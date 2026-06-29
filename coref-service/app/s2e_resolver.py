"""Inference wrapper around the vendored s2e-coref model.

The upstream repo only ships dataset-level evaluation (jsonlines + the CoNLL
scorer); it has no "raw text -> clusters" entry point. This module adds one:

  1. word-tokenise the text while tracking each word's character span,
  2. sub-tokenise with the Longformer tokenizer the way s2e's ``data.py`` does,
  3. run the S2E model with ``return_all_outputs=True``,
  4. decode antecedents into clusters exactly like s2e's ``eval.py``,
  5. map the predicted token spans back to character offsets.

It returns clusters as ``[[(char_start, char_end), ...], ...]`` — the same shape
the LingMess path produces — so the two can be merged uniformly.

The model weights are NOT bundled (the trained checkpoint is ~1.6 GB and lives on
Dropbox, see README). The resolver is only activated when ``S2E_MODEL_PATH``
points at a directory containing the checkpoint; otherwise the caller falls back
to LingMess-only.
"""

from __future__ import annotations

import logging
import os
import re
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# s2e eval-time hyper-parameters (must match the released checkpoint; see the
# project README's evaluation command).
S2E_ARGS = SimpleNamespace(
    max_span_length=30,
    top_lambda=0.4,
    ffnn_size=3072,
    dropout_prob=0.3,
    normalise_loss=True,
)

DEFAULT_TOKENIZER = os.getenv("S2E_TOKENIZER", "allenai/longformer-large-4096")
# Longformer attention works in windows of 512; keep documents within one window
# set to avoid the global-attention bookkeeping the batched eval path handles.
MAX_TOKENS = int(os.getenv("S2E_MAX_TOKENS", "4096"))

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _word_spans(text: str):
    """Return [(word, char_start, char_end), ...] splitting words and punctuation."""
    return [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


class S2EResolver:
    """Loads the s2e checkpoint lazily and resolves text into char-span clusters."""

    def __init__(self, model_path: str | None = None, device: str = "cpu",
                 tokenizer_name: str = DEFAULT_TOKENIZER):
        self.model_path = model_path or os.getenv("S2E_MODEL_PATH", "")
        self.device = device
        self.tokenizer_name = tokenizer_name
        self._model = None
        self._tokenizer = None

    @property
    def configured(self) -> bool:
        return bool(self.model_path) and os.path.isdir(self.model_path)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is not None:
            return
        if not self.configured:
            raise FileNotFoundError(
                f"S2E_MODEL_PATH is not set or not a directory: {self.model_path!r}"
            )
        import torch
        from transformers import LongformerConfig, LongformerTokenizerFast

        from .s2e.modeling import S2E

        logger.info("Loading s2e-coref from %s on %s", self.model_path, self.device)
        config = LongformerConfig.from_pretrained(self.model_path)
        # Longformer has no SDPA path; newer transformers (>=4.36) require asking
        # for "eager" explicitly. Harmless no-op on the pinned transformers 4.30.2.
        config._attn_implementation = "eager"
        model = S2E(config=config, args=S2E_ARGS)

        state_path = os.path.join(self.model_path, "pytorch_model.bin")
        state = torch.load(state_path, map_location=self.device)
        # strict=False tolerates buffer-only diffs (e.g. position_ids) between the
        # checkpoint's transformers version and the pinned one.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("s2e: %d missing state keys (e.g. %s)", len(missing), missing[:3])
        if unexpected:
            logger.warning("s2e: %d unexpected state keys (e.g. %s)", len(unexpected), unexpected[:3])

        model.to(self.device).eval()
        self._model = model
        self._tokenizer = LongformerTokenizerFast.from_pretrained(self.tokenizer_name)

    # ------------------------------------------------------------------ #
    def _encode(self, text: str):
        """Mirror s2e's data.py tokenisation for a single, speaker-less document.

        Returns (input_ids_tensor, attention_mask_tensor, token_idx_to_word_idx,
        word_spans). ``token_idx_to_word_idx`` is indexed by position in the model
        sequence (index 0 == the <s> special token).
        """
        import torch

        words = _word_spans(text)
        token_ids = []
        token_idx_to_word_idx = [0]  # for the leading <s>
        for idx, (word, _s, _e) in enumerate(words):
            sub = self._tokenizer.encode(" " + word, add_special_tokens=False)
            token_ids.extend(sub)
            token_idx_to_word_idx.extend([idx] * len(sub))

        encoded = self._tokenizer.encode_plus(
            token_ids,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
            token_idx_to_word_idx,
            words,
        )

    def clusters(self, text: str) -> list:
        """Resolve ``text`` into clusters of character spans."""
        if not text or not text.strip():
            return []
        self.load()
        import numpy as np
        import torch

        input_ids, attention_mask, tok2word, words = self._encode(text)
        n_words = len(words)
        seq_len = input_ids.size(1)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_all_outputs=True,
            )
        # forward(return_all_outputs=True, gold_clusters=None) ->
        #   (mention_start_ids, mention_end_ids, final_logits, mention_logits)
        starts = outputs[0][0].cpu().numpy()          # [k]
        end_offsets = outputs[1][0].cpu().numpy()     # [k]
        coref_logits = outputs[2][0].cpu().numpy()    # [k, k+1]

        # Decode antecedents (identical logic to s2e eval.py).
        max_antecedents = np.argmax(coref_logits, axis=1).tolist()
        mention_to_antecedent = {
            ((int(s), int(e)), (int(starts[a]), int(end_offsets[a])))
            for s, e, a in zip(starts, end_offsets, max_antecedents)
            if a < len(starts)
        }
        from .s2e.utils import extract_clusters_for_decode

        token_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)

        # Map token spans -> character spans via the word index.
        char_clusters = []
        for cluster in token_clusters:
            spans = []
            for (t_start, t_end) in cluster:
                if t_start >= seq_len or t_end >= seq_len:
                    continue
                w_start = tok2word[t_start]
                w_end = tok2word[t_end]
                if w_start >= n_words or w_end >= n_words:
                    continue
                char_start = words[w_start][1]
                char_end = words[w_end][2]
                if char_end > char_start:
                    spans.append((char_start, char_end))
            # de-dup while preserving order
            seen = set()
            spans = [s for s in spans if not (s in seen or seen.add(s))]
            if len(spans) >= 2:
                char_clusters.append(spans)
        return char_clusters
