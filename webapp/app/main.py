"""FastAPI app: a viewer for the bio-semantic-parser ingestion pipeline.

Serves a single-page UI (``webapp/static``) plus a small JSON API that runs the
real pipeline components and returns each step's output for visualisation.

Run:
    uvicorn webapp.app.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.registry.registry import SourceRegistry
from webapp.app.pipeline_runner import DEFAULT_COREF_URL, PipelineRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
CONFIG_PATH = os.getenv("SOURCES_CONFIG", str(REPO_ROOT / "config" / "sources.yaml"))

app = FastAPI(title="Bio-Semantic-Parser — Pipeline Viewer", version="1.0.0")

runner = PipelineRunner(coref_url=DEFAULT_COREF_URL)


def _load_sources() -> list:
    try:
        return SourceRegistry(CONFIG_PATH).get_all_sources()
    except Exception:
        return []


# ── API models ────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    mode: str = "text"  # "text" or "url"
    text: str | None = None
    url: str | None = None
    source_name: str = "manual"
    text_field: str | None = None
    document_id: str | None = None
    use_coref: bool = True


# ── API routes ────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "coref": runner.coref_status()}


@app.get("/api/sources")
def sources():
    items = []
    for s in _load_sources():
        items.append(
            {
                "name": s.get("name"),
                "type": s.get("type"),
                "format": s.get("format"),
                "base_url": s.get("base_url"),
                "text_field": s.get("text_field"),
            }
        )
    return {"sources": items}


@app.post("/api/run")
def run(req: RunRequest):
    if req.mode == "url":
        if not req.url:
            raise HTTPException(status_code=400, detail="`url` is required for mode=url")
        target = req.url
    else:
        if not (req.text and req.text.strip()):
            raise HTTPException(status_code=400, detail="`text` is required for mode=text")
        target = None

    try:
        return runner.run(
            text=req.text,
            url=target,
            source_name=req.source_name or "manual",
            text_field=req.text_field,
            document_id=req.document_id,
            use_coref=req.use_coref,
        )
    except Exception as exc:  # surface pipeline errors to the UI cleanly
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc


# ── Static UI ─────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
