# Coreference Resolution Service

An HTTP service that rewrites pronouns and references to the entity they point to,
so every sentence stands on its own. It powers step 4 of the parsing pipeline
(`CorefClient`) but runs independently.

```
in:  "Rapamycin was given daily. It reduced mTOR activity."
out: "Rapamycin was given daily. Rapamycin reduced mTOR activity."
```

It offers three models, all on the same API: **`lingmess`** (default, best
accuracy), **`fcoref`** (faster), and **`cascade`** (LingMess first, then
s2e-coref fills the mentions LingMess missed — never overriding it).

## Run — default (LingMess)

```bash
cd coref-service
make up          # build + run on http://localhost:5000  (or: docker compose up -d --build)
make health      # check it's alive
make resolve     # resolve a sample sentence
```

API docs: <http://localhost:5000/docs>

## Run — cascade (LingMess → s2e-coref)

Needs a one-time s2e checkpoint (~1.6 GB). `make cascade` downloads it, then runs:

```bash
make cascade     # downloads checkpoint if missing, starts in cascade mode
make health      # confirm "s2e_active": true
```

Without `make`:

```bash
bash scripts/download_s2e.sh
COREF_MODEL=cascade S2E_MODEL_PATH=/models/s2e docker compose up -d --build
```

If the checkpoint is absent, cascade logs a warning and **falls back to LingMess
only** — `s2e_active` in `/health` shows which is live.

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
| `COREF_MODEL` | `lingmess` | `lingmess`, `fcoref`, or `cascade` |
| `COREF_DEVICE` | `cpu` | `cpu` or `cuda:0` |
| `COREF_RESOLVE_MODE` | `anaphora` | rewrite pronouns and `the X` references, or `pronouns_only` |
| `COREF_PRELOAD` | `true` | load the model at startup |
| `S2E_MODEL_PATH` | _(unset)_ | cascade only: dir with `pytorch_model.bin` + `config.json` |

## Notes

- s2e-coref has no pip package, so its inference code is vendored under `app/s2e/`
  (MIT, see `app/s2e/LICENSE`); only the checkpoint is downloaded separately.
- Tests run offline (no model needed): `pip install -r requirements-dev.txt && make test`.
- Models are trained on general English, not biomedical text — domain tuning is a
  possible next step.
