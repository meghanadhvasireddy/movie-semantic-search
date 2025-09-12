from typing import List, Optional
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search string")
    k: int = Field(5, ge=1, le=50, description="Top-K results")
    page: int = Field(1, ge=1, description="Page number (for future use)")
    per_page: int = Field(5, ge=1, le=50, description="Page size (for future use)")

class SearchResult(BaseModel):
    id: int
    title: str
    snippet: str
    score: float

class SearchResponse(BaseModel):
    query: str
    took_ms: int
    cached: bool  # always False today; True/False starting Day 6
    results: List[SearchResult]

class IndexStats(BaseModel):
    model: str
    dim: int
    doc_count: int
    normalized: bool
    cache_hit_rate: Optional[float] = None  # filled on Day 6
