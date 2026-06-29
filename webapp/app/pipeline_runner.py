"""Instrumented runner for the bio-semantic-parser ingestion pipeline.

This module re-uses the *production* pipeline components (the same classes the
``Fetcher`` uses) and runs them step-by-step while capturing the intermediate
output of every step. The web UI consumes the structured result so a human can
*see* what each layer of the pipeline does to a document.

It deliberately mirrors ``src/fetcher/fetcher.py`` so the viewer reflects the
real behaviour rather than a re-implementation.
"""

from __future__ import annotations

import hashlib
import os
import re
import time

import requests

from src.fetcher.cleaner import TextCleaner
from src.fetcher.coref_client import CorefClient
from src.fetcher.format_detector import FormatDetector
from src.fetcher.metadata_attacher import MetadataAttacher
from src.fetcher.section_splitter import SectionSplitter

DEFAULT_COREF_URL = os.getenv("COREF_URL", "http://localhost:5000")
PREVIEW_CHARS = 1200


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Trim long text for transport to the browser."""
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n… (+{len(text) - limit:,} more chars)"


def _find_coref_rewrites(before: str, after: str) -> list:
    """Return (before_sentence, after_sentence) pairs that the coref step changed.

    Mirrors ``src.fetcher.fetcher._find_coref_rewrites``.
    """
    before_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", before) if s.strip()]
    after_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", after) if s.strip()]
    rewrites = []
    for b, a in zip(before_sents, after_sents):
        if b != a:
            rewrites.append({"before": b, "after": a})
    return rewrites


class PipelineRunner:
    """Runs the ingestion pipeline and records every step for visualisation."""

    def __init__(self, coref_url: str | None = None, timeout: int | None = None):
        self.coref_url = coref_url or DEFAULT_COREF_URL
        self.timeout = timeout or int(os.getenv("REQUEST_TIMEOUT", "30"))

        self.cleaner = TextCleaner()
        self.coref_client = CorefClient(self.coref_url)
        self.section_splitter = SectionSplitter()
        self.metadata_attacher = MetadataAttacher()
        self.format_detector = FormatDetector()
        self._chunker = None  # lazy — Chunker() may download a tiktoken encoding

    # ------------------------------------------------------------------ #
    @property
    def chunker(self):
        if self._chunker is None:
            from src.fetcher.chunker import Chunker

            self._chunker = Chunker()
        return self._chunker

    def coref_status(self) -> dict:
        return {"url": self.coref_url, "online": self.coref_client.health_check()}

    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        text: str | None = None,
        url: str | None = None,
        source_name: str = "manual",
        text_field: str | None = None,
        document_id: str | None = None,
        use_coref: bool = True,
    ) -> dict:
        """Execute the pipeline and return a structured, per-step report."""
        started = time.time()
        steps: list[dict] = []

        # ── Step 1 — acquire raw text ─────────────────────────────────
        if url:
            t0 = time.time()
            resp = requests.get(url, timeout=self.timeout)
            content_type = resp.headers.get("Content-Type", "")
            fmt = self.format_detector.detect(content_type)
            raw = self._extract_text(fmt, resp, text_field)
            steps.append(
                {
                    "id": 1,
                    "name": "Fetch & extract",
                    "status": "ok",
                    "summary": (
                        f"HTTP {resp.status_code} · {content_type or 'unknown'} · "
                        f"format={fmt.upper()} · {len(raw):,} chars extracted"
                    ),
                    "detail": {
                        "url": url,
                        "http_status": resp.status_code,
                        "content_type": content_type,
                        "format": fmt,
                        "bytes": len(resp.content),
                    },
                    "content": _preview(raw),
                }
            )
        else:
            raw = text or ""
            fmt = "text"
            steps.append(
                {
                    "id": 1,
                    "name": "Input",
                    "status": "ok",
                    "summary": f"Pasted text · {len(raw):,} chars",
                    "detail": {"format": fmt},
                    "content": _preview(raw),
                }
            )

        if document_id is None:
            document_id = hashlib.sha256((raw or source_name).encode("utf-8")).hexdigest()

        # ── Step 2 — clean noise ──────────────────────────────────────
        before_len = len(raw)
        cleaned = self.cleaner.clean(raw)
        steps.append(
            {
                "id": 2,
                "name": "Clean noise",
                "status": "ok",
                "summary": (
                    f"{before_len:,} → {len(cleaned):,} chars "
                    f"({before_len - len(cleaned):,} removed)"
                ),
                "detail": {
                    "chars_before": before_len,
                    "chars_after": len(cleaned),
                    "chars_removed": before_len - len(cleaned),
                },
                "content": _preview(cleaned),
            }
        )

        # ── Step 3 — coreference resolution ───────────────────────────
        resolved = cleaned
        if not use_coref:
            steps.append(
                {
                    "id": 3,
                    "name": "Coreference resolution",
                    "status": "skipped",
                    "summary": "Disabled for this run",
                    "detail": {"coref_url": self.coref_url},
                    "content": _preview(resolved),
                }
            )
        elif self.coref_client.health_check():
            resolved = self.coref_client.resolve(cleaned)
            rewrites = _find_coref_rewrites(cleaned, resolved)
            steps.append(
                {
                    "id": 3,
                    "name": "Coreference resolution",
                    "status": "ok",
                    "summary": (
                        f"Service ONLINE · {len(rewrites)} sentence(s) rewritten"
                        if rewrites
                        else "Service ONLINE · no pronouns resolved"
                    ),
                    "detail": {"coref_url": self.coref_url, "rewrites": rewrites},
                    "content": _preview(resolved),
                }
            )
        else:
            steps.append(
                {
                    "id": 3,
                    "name": "Coreference resolution",
                    "status": "offline",
                    "summary": f"Service OFFLINE ({self.coref_url}) · text passed through unchanged",
                    "detail": {"coref_url": self.coref_url},
                    "content": _preview(resolved),
                }
            )

        # ── Step 4 — section splitting ────────────────────────────────
        sections = self.section_splitter.split(resolved)
        steps.append(
            {
                "id": 4,
                "name": "Section splitting",
                "status": "ok",
                "summary": f"{len(sections)} section(s): "
                + ", ".join(s["section"] for s in sections),
                "detail": {
                    "sections": [
                        {"section": s["section"], "chars": len(s["text"]), "preview": _preview(s["text"], 300)}
                        for s in sections
                    ]
                },
            }
        )

        # ── Step 5 — chunking ─────────────────────────────────────────
        try:
            chunks = self.chunker.chunk_document(sections)
            chunk_status = "ok"
            chunk_summary = f"{len(chunks)} chunk(s) produced"
            chunk_error = None
        except Exception as exc:  # tiktoken download can fail offline
            chunks = [
                {"text": s["text"], "section": s["section"], "chunk_index": 0, "total_chunks": 1}
                for s in sections
            ]
            chunk_status = "error"
            chunk_summary = f"Chunker unavailable ({exc}) · fell back to 1 chunk/section"
            chunk_error = str(exc)
        steps.append(
            {
                "id": 5,
                "name": "Chunking",
                "status": chunk_status,
                "summary": chunk_summary,
                "detail": {"count": len(chunks), "error": chunk_error},
            }
        )

        # ── Step 6 — metadata attachment ──────────────────────────────
        enriched = self.metadata_attacher.attach(chunks, document_id, source_name, url or "")
        steps.append(
            {
                "id": 6,
                "name": "Metadata attachment",
                "status": "ok",
                "summary": f"Attached metadata to {len(enriched)} chunk(s)",
                "detail": {
                    "document_id": document_id,
                    "source_name": source_name,
                    "source_url": url or "",
                },
            }
        )

        return {
            "ok": True,
            "document_id": document_id,
            "source_name": source_name,
            "elapsed_ms": round((time.time() - started) * 1000),
            "steps": steps,
            "chunks": enriched,
        }

    # ------------------------------------------------------------------ #
    def _extract_text(self, fmt: str, response, text_field: str | None) -> str:
        # Imported lazily — handlers pull in bs4/pdf libs only needed for URL mode.
        from src.fetcher.handlers import HTMLHandler, JSONHandler, PDFHandler, XMLHandler

        if fmt == "json":
            return JSONHandler().extract(response.text, text_field)
        if fmt == "xml":
            return XMLHandler().extract(response.text, text_field)
        if fmt == "html":
            return HTMLHandler().extract(response.text)
        if fmt == "pdf":
            return PDFHandler().extract(response.content)
        return response.text
