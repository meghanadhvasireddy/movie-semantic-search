# app/main.py
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator

from app import config
from app.models import IndexStats, SearchRequest, SearchResponse, SearchResult
from app.search_service import SearchService

# ---------- Logging: JSON lines ----------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("app")

# ---------- Lifespan: initialize/teardown singletons ----------
service: Optional[SearchService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    # initialize the service (use REDIS_URL from config / env)
    service = SearchService(redis_url=config.REDIS_URL)
    try:
        yield
    finally:
        service = None


# ---------- FastAPI app ----------
app = FastAPI(title="Semantic Search API", version="0.1.0", lifespan=lifespan)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# Access-log middleware (timing every request)
@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    took_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        json.dumps(
            {
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status": resp.status_code,
                "took_ms": took_ms,
            }
        )
    )
    return resp


# ---------- Endpoints ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "message": "service running"}


@app.get("/index/stats", response_model=IndexStats)
def index_stats():
    assert service is not None
    return service.stats()


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    assert service is not None
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    t0 = time.perf_counter()
    took_ms_inner, items, cached = service.search(q, k=req.k, page=req.page, per_page=req.per_page)
    total_ms = int((time.perf_counter() - t0) * 1000)

    # structured search event
    log.info(
        json.dumps(
            {
                "event": "search",
                "query_len": len(q),
                "k": req.k,
                "page": req.page,
                "per_page": req.per_page,
                "cached": cached,
                "svc_took_ms": took_ms_inner,
                "endpoint_took_ms": total_ms,
            }
        )
    )

    return SearchResponse(
        query=q,
        took_ms=total_ms,  # full endpoint timing
        cached=cached,
        results=[SearchResult(**it) for it in items],
    )
