# Coreference Resolution Service

A small, self-contained HTTP service that performs **coreference resolution**:
it rewrites pronouns and references to the entity they point to, so every
sentence stands on its own.

```
in:  "Rapamycin was given daily. It reduced mTOR activity."
out: "Rapamycin was given daily. Rapamycin reduced mTOR activity."
```

It powers step 4 of the parsing pipeline (`CorefClient` in the main project), but
runs as an independent service with a tiny JSON API.

---

## Quickstart

```bash
cd coref-service
make up            # build + run LingMess on http://localhost:5000
make health        # check it's alive
make resolve       # resolve a sample sentence
```

No `make`? The plain equivalents:

```bash
docker compose up -d --build
curl localhost:5000/health
```

Interactive API docs are served at <http://localhost:5000/docs>.

---

## Choosing a model

Set `COREF_MODEL` (env or `.env`). All three speak the same API.

| `COREF_MODEL` | Model | Use it when |
|---|---|---|
| `lingmess` *(default)* | LingMess (via `fastcoref`) | Best accuracy, zero setup — the right default. |
| `fcoref` | F-coref (via `fastcoref`) | You want a faster, lighter model. |
| `cascade` | LingMess **→** s2e-coref | You want maximum recall: s2e catches mentions LingMess missed. Needs a one-time checkpoint download. |

### Background: s2e-coref vs LingMess

Both are neural models with a Longformer encoder, evaluated on OntoNotes.

| | s2e-coref | LingMess |
|---|---|---|
| Approach | Scores antecedents directly from token endpoints (low memory) | Separate expert scorers per mention-pair type |
| OntoNotes F1 | ~80.3 | ~81.4 |
| Packaging | Research repo, manual setup | `fastcoref` pip package |

LingMess scores higher and installs cleanly, so it's the default. The **cascade**
mode below combines them.

> Note: both models are trained on general English, not biomedical text. Tuning
> for scientific text is a possible next step.

---

## Cascade mode (LingMess → s2e-coref)

The two models make different mistakes, so cascade runs them in sequence:
**LingMess resolves first and stays authoritative; s2e-coref only *adds*
resolutions for anaphors LingMess left unlinked** — it never overrides a LingMess
decision (see `_merge_clusters` in `app/resolver.py`).

Because s2e-coref ships as a research repo with no pip package, its model code is
vendored under `app/s2e/` (MIT-licensed, see `app/s2e/LICENSE`) — only the
inference pieces, with imports patched for the pinned transformers. The trained
checkpoint (~1.6 GB) is downloaded separately.

**One command:**

```bash
make cascade       # downloads the checkpoint if needed, then starts in cascade mode
make health        # confirm "s2e_active": true
```

Or step by step / without `make`:

```bash
make download-s2e                      # or: bash scripts/download_s2e.sh
COREF_MODEL=cascade S2E_MODEL_PATH=/models/s2e docker compose up -d --build
```

If the checkpoint is missing, cascade **falls back to LingMess only** (logs a
warning, keeps serving). `s2e_active` in `/health` tells you which is live.

---

## API

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/health` | — | service + model status |
| `POST` | `/resolve` | `{"text": "..."}` | `{"resolved_text": "..."}` |
| `POST` | `/clusters` | `{"text": "..."}` | coreference clusters with character offsets |

```bash
curl -X POST localhost:5000/resolve -H 'Content-Type: application/json' \
  -d '{"text":"Rapamycin reduced mTOR. It was effective. The drug helped."}'
# {"resolved_text":"Rapamycin reduced mTOR. Rapamycin was effective. Rapamycin helped."}
```

`/health` in cascade mode:

```json
{"status":"ok","model":"cascade","cascade":true,"s2e_active":true, ...}
```

---

## Configuration

Set via your shell or a `.env` file (docker compose reads it; see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `COREF_MODEL` | `lingmess` | `lingmess`, `fcoref`, or `cascade` |
| `COREF_DEVICE` | `cpu` | `cpu` or `cuda:0` |
| `COREF_RESOLVE_MODE` | `anaphora` | rewrite pronouns and `the X` references, or `pronouns_only` |
| `COREF_PRELOAD` | `true` | load the model at startup (vs first request) |
| `S2E_MODEL_PATH` | _(unset)_ | cascade only: dir with `pytorch_model.bin` + `config.json` |
| `S2E_TOKENIZER` | `allenai/longformer-large-4096` | cascade only: tokenizer/config for s2e |

---

## Development

```bash
pip install -r requirements-dev.txt
make test          # or: pytest -q
```

The tests cover the cascade merge, the rewrite logic, and the s2e tokenisation
**without** loading any model (the heavy imports live inside `load()`), so they
run fast and offline.

Project layout:

```
app/
  main.py          FastAPI app (/health, /resolve, /clusters)
  resolver.py      LingMess/F-coref + cascade merge + text rewriting
  s2e_resolver.py  s2e-coref inference wrapper (text → char-span clusters)
  s2e/             vendored s2e-coref model code (inference only, MIT)
scripts/
  download_s2e.sh  fetch the s2e checkpoint
tests/             unit tests (no model required)
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `s2e_active: false` in cascade mode | Checkpoint missing — run `make download-s2e`, confirm `s2e-model/pytorch_model.bin` exists, restart. |
| First start is slow | Model weights download on first run; cached in the `coref-models` volume afterwards. |
| Slow Hugging Face downloads | The image sets `HF_HUB_DISABLE_XET=1`; keep it. |
| Out of memory in cascade | s2e is a large Longformer model. Use `COREF_DEVICE=cuda:0`, or stick with `lingmess`. |
