# Pipeline Viewer

A small web app to **see the bio-semantic-parser ingestion pipeline work**.

Paste a biomedical abstract (or point it at a URL) and watch the document flow
through every step of the pipeline — fetch/extract → clean → coreference
resolution → section splitting → chunking → metadata attachment — with the
intermediate output of each step shown in the browser.

It drives the **real** pipeline classes from `src/` (the same ones `Fetcher`
uses), so what you see is the actual behaviour, not a mock.

## What it shows

| Step | Pipeline component | What you see |
|------|--------------------|--------------|
| 1 | `FormatDetector` + handlers | raw input / fetched & extracted text |
| 2 | `TextCleaner` | chars removed, cleaned text |
| 3 | `CorefClient` | online/offline, the sentences that were rewritten |
| 4 | `SectionSplitter` | detected sections + per-section preview |
| 5 | `Chunker` | number of chunks produced |
| 6 | `MetadataAttacher` | document_id, source, position on every chunk |

The header shows live **coreference-service health** (polled every 15 s).

## Run

From the **repository root**:

```bash
# 1. install deps (a virtualenv is recommended)
pip install -r webapp/requirements.txt
#    plus the pipeline's own deps if not already installed:
pip install pyyaml requests python-dotenv tiktoken

# 2. start the app
uvicorn webapp.app.main:app --reload --port 8000
```

Open <http://localhost:8000>.

> The coreference step is optional. If no coreference service is running at
> `COREF_URL`, the viewer shows it as **offline** and passes the text through
> unchanged — exactly like the real pipeline (`Fetcher` step 4) does.

### Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `COREF_URL` | `http://localhost:5000` | coreference service base URL |
| `SOURCES_CONFIG` | `config/sources.yaml` | source registry to populate the dropdown |
| `REQUEST_TIMEOUT` | `30` | HTTP timeout (seconds) for URL fetch / coref |

## API

The UI is a thin client over a small JSON API:

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET` | `/api/health` | – | service + coref status |
| `GET` | `/api/sources` | – | registered sources from `config/sources.yaml` |
| `POST` | `/api/run` | `{mode, text?, url?, source_name?, text_field?, use_coref?}` | per-step report + output chunks |

Example:

```bash
curl -s localhost:8000/api/run -H 'content-type: application/json' \
  -d '{"mode":"text","text":"Rapamycin was given. It reduced mTOR activity.","use_coref":false}' | jq .steps
```

## Tests

```bash
cd webapp
pip install -r requirements-dev.txt
pytest
```

The tests run fully offline (coreference disabled, chunker degrades gracefully).
