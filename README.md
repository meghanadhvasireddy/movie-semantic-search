Semantic Search API

A production-grade **semantic search microservice** for querying movie plots by meaning, not just keywords.  
Built with FastAPI, FAISS, and SentenceTransformers, with Redis caching, Prometheus monitoring, Docker deployment, and full testing support.

---

Features
- Semantic search with embeddings (`all-MiniLM-L6-v2`) + FAISS
- Hybrid reranking (semantic + TF-IDF) for sharper results
- Redis caching with in-memory fallback
- REST API with [Swagger UI](http://127.0.0.1:8000/docs) + [OpenAPI schema](docs/openapi.json)
- Prometheus metrics at `/metrics`
- Docker & Compose setup for one-command deployment
- Smoke test script + pytest integration tests

---

## Quickstart

### Local Setup
```bash
git clone https://github.com/<your-username>/semantic-search.git
cd semantic-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Docker
```bash
docker compose up --build
```
Runs both the API and Redis cache.

---

## Example Usage

**Health check**
```bash
curl http://127.0.0.1:8000/healthz
```
```json
{"status":"ok","message":"service running"}
```

**Index stats**
```bash
curl http://127.0.0.1:8000/index/stats | jq .
```

**Search query**
```bash
curl http://127.0.0.1:8000/search \
  -H "content-type: application/json" \
  -d '{"query":"astronaut stranded on Mars","k":5,"page":1,"per_page":5}' | jq .
```

Sample response:
```json
{
  "query": "astronaut stranded on Mars",
  "took_ms": 24,
  "cached": false,
  "results": [
    {"id": 101, "title": "Robinson Crusoe on Mars", "snippet": "An astronaut..."}
  ]
}
```

---

## Testing

Run full test suite:
```bash
pytest -q
```

Smoke test:
```bash
./scripts/smoke.sh
```

---

## Configuration

Environment variables (`.env.example` included):

```
REDIS_URL=redis://localhost:6379/0
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
SEMANTIC_K=200
HYBRID_ALPHA=0.8
```

---

## Tech Stack
- FastAPI
- SentenceTransformers
- FAISS
- Redis
- Docker & Docker Compose
- pytest
- Prometheus FastAPI Instrumentator


## Author
**Meghanadh Vasireddy**  
 Brown University (CS & Applied Math)
[GitHub](https://github.com/meghanadhvasireddy) · [LinkedIn](https://www.linkedin.com/in/meghanadhv/)
