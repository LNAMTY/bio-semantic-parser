# Coreference Resolution Service

A small local service that performs coreference resolution. It replaces pronouns
and references with the entity they point to, so each sentence stands on its own.

```
in:  "Rapamycin was given daily. It reduced mTOR activity."
out: "Rapamycin was given daily. Rapamycin reduced mTOR activity."
```

## Model comparison: s2e-coref vs LingMess

Both are neural coreference models that use a Longformer encoder and are evaluated
on the OntoNotes benchmark.

| | s2e-coref | LingMess |
|---|---|---|
| Approach | Scores antecedents directly from token endpoints (low memory) | Separate expert scorers per mention-pair type |
| OntoNotes F1 | ~80.3 | ~81.4 |
| Packaging | Research repo, manual setup | `fastcoref` pip package |
| Speed option | None | Also ships FCoref, a faster distilled model |

**Chosen: LingMess (via `fastcoref`).** It scores higher, installs as a normal
package, and gives a faster fallback model (`fcoref`) through the same API.

Note: both models are trained on general English, not biomedical text. Tuning for
scientific text is a possible next step.

### Cascade mode (LingMess → s2e-coref)

The two models make different mistakes, so the service can run them in sequence:
**LingMess resolves first, then s2e-coref handles the mentions LingMess missed.**
LingMess's clusters are authoritative — s2e only *adds* resolutions for anaphors
that LingMess left unlinked (it never overrides a LingMess decision). See
`_merge_clusters` in `app/resolver.py`.

Enable it with `COREF_MODEL=cascade`. Because s2e-coref ships as a research repo
with no pip package, its model code is vendored under `app/s2e/` (MIT-licensed,
see `app/s2e/LICENSE`) and the trained checkpoint is supplied separately:

```bash
# download the released s2e checkpoint (~1.6 GB) into ./s2e-model
mkdir -p s2e-model
curl -L "https://www.dropbox.com/sh/7hpw662xylbmi5o/AAC3nfP4xdGAkf0UkFGzAbrja?dl=1" \
  -o s2e.zip && unzip s2e.zip -d s2e-model && rm s2e.zip
```

Then mount it and point the service at it (see `docker-compose.yml`):

```yaml
environment:
  COREF_MODEL: cascade
  S2E_MODEL_PATH: /models/s2e
volumes:
  - ./s2e-model:/models/s2e:ro
```

If `S2E_MODEL_PATH` is unset or the checkpoint is missing, cascade mode logs a
warning and **falls back to LingMess only** — the service still works. Check
`s2e_active` in `/health` to confirm the second stage loaded.

## Run

A prebuilt image is published, so there is no need to build from source:

```bash
cd coref-service
docker compose pull         # pulls ghcr.io/lnamty/coref-service:lingmess
docker compose up -d        # serves on http://localhost:5000
```

The first start downloads the model weights into the `coref-models` volume. Later
starts reuse them.

To build the image locally instead of pulling it:

```bash
docker compose build
docker compose up -d
```

## API

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | | service and model status |
| POST | `/resolve` | `{"text": "..."}` | `{"resolved_text": "..."}` |
| POST | `/clusters` | `{"text": "..."}` | coreference clusters with offsets |

Example:

```bash
curl -X POST localhost:5000/resolve \
  -H 'Content-Type: application/json' \
  -d '{"text":"Rapamycin was given daily. It reduced mTOR activity."}'
```

## Configuration

Set these in `docker-compose.yml` or a `.env` file (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `COREF_MODEL` | `lingmess` | `lingmess` (accurate), `fcoref` (fast), or `cascade` (LingMess → s2e) |
| `COREF_DEVICE` | `cpu` | `cpu` or `cuda:0` |
| `COREF_RESOLVE_MODE` | `anaphora` | rewrite pronouns and `the X` references, or `pronouns_only` |
| `COREF_PRELOAD` | `true` | load the model at startup |
| `S2E_MODEL_PATH` | _(unset)_ | cascade only: dir with the s2e checkpoint (`pytorch_model.bin` + `config.json`) |
| `S2E_TOKENIZER` | `allenai/longformer-large-4096` | cascade only: tokenizer/config for s2e |
