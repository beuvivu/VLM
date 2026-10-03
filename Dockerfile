# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------- build stage
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".[crawler]"

# -------------------------------------------------------------- runtime stage
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    VQE_DATA_DIR=/app/data \
    VQE_SEED_FILE_DIR=/app/seed \
    VQE_PRODUCT_SEED_DIR=/app/seed \
    VQE_CALIBRATION_DIR=/app/calibration \
    VQE_ANCHORS_FILE=/app/seed/jackpot_anchors.json \
    VQE_LOG_JSON=true
WORKDIR /app
RUN useradd --create-home --uid 10001 vqe
COPY --from=builder /opt/venv /opt/venv
# The bundled data live outside /app/data so the data volume (DuckDB, forecaster state,
# ledger) never hides them.
COPY data/seed ./seed
COPY data/calibration ./calibration
COPY scripts ./scripts
RUN mkdir -p /app/data /app/reports && chown -R vqe:vqe /app
USER vqe

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker: DuckDB allows a single read-write process per database file.
# CPU-heavy endpoints are sync functions and run in FastAPI's thread pool.
CMD ["uvicorn", "vietlott_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
