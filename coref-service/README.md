# Coreference Resolution Service

An HTTP service that rewrites pronouns and references to the entity they point to,
so every sentence stands on its own. It powers step 4 of the parsing pipeline
(`CorefClient`) but runs independently.

```
in:  "Rapamycin was given daily. It reduced mTOR activity."
out: "Rapamycin was given daily. Rapamycin reduced mTOR activity."
```

The default model is **`cascade`**: LingMess resolves first, then s2e-coref fills
the mentions LingMess missed (never overriding it). Two lighter alternatives use
the same API — **`lingmess`** (LingMess only) and **`fcoref`** (faster).

## Run

```bash
cd coref-service
make up          # build + run on http://localhost:5000  (or: docker compose up -d --build)
make health      # check it's alive  ("s2e_active": true in cascade)
make resolve     # resolve a sample sentence
```

On the **first** cascade start the s2e checkpoint (~1.6 GB) is downloaded
automatically into the `coref-models` volume; later starts reuse it (so the first
boot is slow — watch `make logs`). If the download fails it logs a warning and
**falls back to LingMess only**; `s2e_active` in `/health` shows which is live.

To run the lighter LingMess-only service and skip the download:

```bash
COREF_MODEL=lingmess docker compose up -d --build
```

API docs: <http://localhost:5000/docs>

## API

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/health` | — | service + model status |
| `POST` | `/resolve` | `{"text": "..."}` | `{"resolved_text": "..."}` |
| `POST` | `/clusters` | `{"text": "..."}` | clusters with character offsets |

```bash
curl -X POST localhost:5000/resolve -H 'Content-Type: application/json' \
  -d '{"text":"Rapamycin reduced mTOR. It was effective. The drug helped."}'
# {"resolved_text":"Rapamycin reduced mTOR. Rapamycin was effective. Rapamycin helped."}
```

## Configuration

Set via your shell or a `.env` file (compose reads it; see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `COREF_MODEL` | `cascade` | `cascade`, `lingmess`, or `fcoref` |
| `COREF_DEVICE` | `cpu` | `cpu` or `cuda:0` |
| `COREF_RESOLVE_MODE` | `anaphora` | rewrite pronouns and `the X` references, or `pronouns_only` |
| `COREF_PRELOAD` | `true` | load the model at startup |
| `S2E_MODEL_PATH` | `/models/s2e` | cascade only: dir with the s2e checkpoint |
| `S2E_AUTO_DOWNLOAD` | `true` | cascade only: fetch the checkpoint on first start |

## Notes

- s2e-coref has no pip package, so its inference code is vendored under `app/s2e/`
  (MIT, see `app/s2e/LICENSE`); only the checkpoint is fetched at runtime.
- Tests run offline (no model needed): `pip install -r requirements-dev.txt && make test`.
- Models are trained on general English, not biomedical text — domain tuning is a
  possible next step.
