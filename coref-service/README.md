# Coreference Resolution Service

Rewrites pronouns and references to the entity they point to, so each sentence
stands on its own. Powers step 4 of the pipeline (`CorefClient`) but runs
independently.

```
in:  "Rapamycin was given daily. It reduced mTOR activity."
out: "Rapamycin was given daily. Rapamycin reduced mTOR activity."
```

Default model is **`cascade`** (LingMess first, then s2e-coref fills the mentions
LingMess missed). Alternatives via the same API: `lingmess`, `fcoref`.

## Run

```bash
cd coref-service
make up        # docker compose up -d --build → http://localhost:5000/docs
make health    # "s2e_active": true once ready
make resolve   # resolve a sample
```

In cascade, the s2e checkpoint (~1.9 GB) downloads in the background on first
start: the service runs on LingMess immediately and the s2e stage activates once
the download lands. If it fails, the service stays LingMess-only.

## API

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/health` | — | service + model status |
| `POST` | `/resolve` | `{"text": "..."}` | `{"resolved_text": "..."}` |
| `POST` | `/clusters` | `{"text": "..."}` | clusters with character offsets |

## Configuration

Set via shell or a `.env` file (see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `COREF_MODEL` | `cascade` | `cascade`, `lingmess`, or `fcoref` |
| `COREF_DEVICE` | `cpu` | `cpu` or `cuda:0` |
| `COREF_RESOLVE_MODE` | `anaphora` | `anaphora` or `pronouns_only` |
| `COREF_PRELOAD` | `true` | load the model at startup |
| `S2E_MODEL_PATH` | `/models/s2e` | cascade: dir with the s2e checkpoint |
| `S2E_AUTO_DOWNLOAD` | `true` | cascade: fetch the checkpoint on first start |

## Tests

```bash
pip install -r requirements-dev.txt && make test   # offline, no model needed
```

s2e-coref has no pip package, so its inference code is vendored under `app/s2e/`
(MIT, see `app/s2e/LICENSE`); only the checkpoint is fetched at runtime.
