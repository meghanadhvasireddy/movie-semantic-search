from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    page: int = 1
    per_page: int = 5

    # Day 12 knobs
    use_synonyms: bool = True
    highlight: bool = True

    # Optional filters (only apply if your docs have these fields)
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    genres: Optional[List[str]] = Field(default=None, description="e.g., ['Drama','Sci-Fi']")

class SearchResult(BaseModel):
    id: int
    title: str
    snippet: str
    score: float
    year: Optional[int] = None
    genres: Optional[List[str]] = None

class SearchResponse(BaseModel):
    query: str
    took_ms: int
    cached: bool
    results: List[SearchResult]

class IndexStats(BaseModel):
    model: str
    dim: int
    doc_count: int
    cache_hit_rate: float = 0.0
