#!/usr/bin/env bash
# In cascade mode, make sure the s2e checkpoint is present before the app starts.
# Downloads it once into the cached volume; on failure the service still starts
# and falls back to LingMess-only.
set -e

if [ "${COREF_MODEL}" = "cascade" ] && [ "${S2E_AUTO_DOWNLOAD:-true}" != "false" ]; then
  if [ ! -f "${S2E_MODEL_PATH}/pytorch_model.bin" ]; then
    echo "[entrypoint] cascade mode: s2e checkpoint missing, downloading once..."
    bash /app/scripts/download_s2e.sh "${S2E_MODEL_PATH}" \
      || echo "[entrypoint] s2e download failed; falling back to LingMess-only."
  else
    echo "[entrypoint] s2e checkpoint found at ${S2E_MODEL_PATH}."
  fi
fi

exec "$@"
