import os

# API port inside the container (external mapping controlled by compose)
PORT = int(os.getenv("PORT", "8000"))

# Uvicorn/Gunicorn worker count (sane default = CPU cores)
WORKERS = int(os.getenv("WORKERS", "2"))

# Redis connection (compose sets this for you)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Request/response timeouts (seconds)
GRACEFUL_TIMEOUT = int(os.getenv("GRACEFUL_TIMEOUT", "30"))
KEEPALIVE = int(os.getenv("KEEPALIVE", "5"))
