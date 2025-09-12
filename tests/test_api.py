# tests/test_api.py
import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

pytestmark = pytest.mark.skipif(
    not os.path.exists("artifacts/faiss.index"),
    reason="index not built yet",
)

def test_search_basic():
    # ✅ ensures startup/shutdown events fire
    with TestClient(app) as client:
        r = client.post("/search", json={"query": "space mission", "k": 3, "page": 1, "per_page": 3})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data and len(data["results"]) == 3
