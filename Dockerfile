# Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install Python deps (ensure gunicorn & uvicorn are in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy source (we mount data/artifacts via compose)
COPY app app

EXPOSE 8000

# Health check
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8000/healthz || exit 1

# 3) Start server
CMD gunicorn app.main:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers ${WORKERS:-2} \
  --graceful-timeout ${GRACEFUL_TIMEOUT:-30} \
  --keep-alive ${KEEPALIVE:-5} \
  --timeout 90
