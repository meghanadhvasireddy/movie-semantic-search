from fastapi import FastAPI, HTTPException
from app.models import SearchRequest, SearchResponse, SearchResult, IndexStats
from app.search_service import SearchService
from typing import Optional


app = FastAPI(title="Semantic Search API", version="0.1.0")

# initialize global service singletons at startup
service: Optional[SearchService] = None

@app.on_event("startup")
def _startup():
    global service
    try:
        service = SearchService()
    except Exception as e:
        # If artifacts are missing, it's better to fail fast with clear message
        raise RuntimeError(f"Startup failed: {e}")

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
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    took_ms, items, cached = service.search(q, k=req.k, page=req.page, per_page=req.per_page)
    return SearchResponse(
        query=q,
        took_ms=took_ms,
        cached=cached,
        results=[SearchResult(**it) for it in items]
    )


