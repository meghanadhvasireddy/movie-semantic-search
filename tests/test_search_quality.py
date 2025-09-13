import os, pytest
from fastapi.testclient import TestClient
from app.main import app

pytestmark = pytest.mark.skipif(
    not os.path.exists("artifacts/faiss.index"),
    reason="index not built yet",
)

def test_normalized_cache_and_paging():
    with TestClient(app) as client:
        body1 = {"query":"Astronaut stranded on Mars!!!","k":8,"page":1,"per_page":4}
        r1 = client.post("/search", json=body1).json()
        assert r1["cached"] is False
        assert len(r1["results"]) == 4

        # Same meaning, different punctuation/case → should be cached now
        body2 = {"query":"astronaut stranded on mars","k":8,"page":1,"per_page":4}
        r2 = client.post("/search", json=body2).json()
        assert r2["cached"] is True
        assert len(r2["results"]) == 4

        # Page 2
        body3 = {"query":"astronaut stranded on mars","k":8,"page":2,"per_page":4}
        r3 = client.post("/search", json=body3).json()
        assert len(r3["results"]) == 4
