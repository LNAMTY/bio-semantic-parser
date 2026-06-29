FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    COREF_MODEL=cascade \
    COREF_DEVICE=cpu \
    COREF_RESOLVE_MODE=anaphora \
    COREF_PRELOAD=true \
    HF_HUB_DISABLE_XET=1 \
    S2E_MODEL_PATH=/models/s2e \
    S2E_AUTO_DOWNLOAD=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Slow mirrors can drop the connection mid-download; retry instead of failing the build.
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY app ./app
COPY scripts ./scripts
COPY entrypoint.sh .

EXPOSE 5000

# Long start-period: cascade's first boot may download the s2e checkpoint (~1.6 GB).
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
  CMD curl -fsS http://localhost:5000/health || exit 1

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
