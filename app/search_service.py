import json, os, time
from typing import Dict, Tuple, List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.searcher import FaissSearcher, META_PATH
import redis   ### NEW

DOCS_PATH = "data/processed/docs.jsonl"

class SearchService:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        # load meta
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model_name = self.meta["model"]
        self.model = SentenceTransformer(self.model_name)
        self.searcher = FaissSearcher()
        self.id_to_doc = self._load_docs_map()

        # connect to redis
        self.redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_docs_map(self) -> Dict[int, Dict]:
        m: Dict[int, Dict] = {}
        if not os.path.exists(DOCS_PATH):
            return m
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                mid = int(obj["id"])
                m[mid] = {"title": obj.get("title", f"id_{mid}"),
                          "text":  obj.get("text", "")}
        return m

    def embed_query(self, q: str) -> np.ndarray:
        vec = self.model.encode([q], normalize_embeddings=True)
        return vec.astype("float32")

    def _cache_key(self, query: str, k: int) -> str:
        return f"search:{self.model_name}:{k}:{query.strip().lower()}"

    def search(self, query: str, k: int) -> Tuple[int, List[Dict], bool]:
        """
        Returns (took_ms, results, cached)
        """
        key = self._cache_key(query, k)
        cached_blob = self.redis.get(key)
        if cached_blob:
            self.cache_hits += 1
            results = json.loads(cached_blob)
            return 1, results, True   # ~1ms to pull from cache

        # otherwise compute fresh
        self.cache_misses += 1
        t0 = time.time()
        qvec = self.embed_query(query)
        scores, ids = self.searcher.search_vectors(qvec, k=k)
        took_ms = int((time.time() - t0) * 1000)

        results: List[Dict] = []
        for score, mid in zip(scores[0], ids[0]):
            mid = int(mid)
            meta = self.id_to_doc.get(mid, {"title": f"id_{mid}", "text": ""})
            snippet = (meta["text"][:220] + "…") if len(meta["text"]) > 220 else meta["text"]
            results.append({
                "id": mid,
                "title": meta["title"],
                "snippet": snippet,
                "score": float(score)
            })

        # store in redis (expire after 24h = 86400 sec)
        self.redis.setex(key, 86400, json.dumps(results))
        return took_ms, results, False

    def stats(self) -> Dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "model": self.model_name,
            "dim":  int(self.searcher.dim),
            "doc_count": len(self.searcher.ids),
            "normalized": True,
            "cache_hit_rate": round(hit_rate, 3)
        }
