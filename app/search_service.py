import json
import os
import re
import time
import uuid
from typing import Dict, Tuple, List

import numpy as np
import redis
from redis.exceptions import RedisError
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.searcher import FaissSearcher

DOCS_PATH = "data/processed/docs.jsonl"


def normalize_query(q: str) -> str:
    """Lowercase, strip punctuation (keep spaces), collapse whitespace."""
    q = (q or "").lower()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


class SearchService:
    def __init__(self, redis_url: str):
        # ----- Embedding model -----
        self.model_name = os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.model_name)

        # ----- FAISS searcher (loads artifacts) -----
        # expects .search_vectors(qvec, k) -> (scores, ids), and .dim, .ids
        self.searcher = FaissSearcher()

        # ----- Metadata: id -> {title, text, (optional year, genres)} -----
        self.id_to_doc = self._load_docs_map()

        # ----- Retrieval / rerank knobs -----
        self.semantic_k = int(os.getenv("SEMANTIC_K", "200"))  # candidate pool from FAISS
        self.alpha = float(os.getenv("HYBRID_ALPHA", "0.8"))   # mix semantic vs tfidf

        # ----- Cache accounting -----
        self.cache_hits = 0
        self.cache_misses = 0

        # ----- Cache namespacing (avoid cross-run pollution) -----
        self.cache_ns = os.getenv("CACHE_NAMESPACE") or str(uuid.uuid4())

        # ----- Redis (optional). If unavailable, fall back to in-memory cache -----
        self.redis_ok = False
        try:
            self.redis = redis.Redis.from_url(redis_url, decode_responses=False)
            self.redis.ping()
            self.redis_ok = True
        except Exception:
            self.redis = None
            self.redis_ok = False

        # In-memory fallback cache: key -> (expires_at, bytes)
        self._mem: Dict[str, Tuple[float, bytes]] = {}
        self._mem_ttl = 86400  # 24h

    # ---------------------- data loading ----------------------

    def _load_docs_map(self) -> Dict[int, Dict]:
        m: Dict[int, Dict] = {}
        if os.path.exists(DOCS_PATH):
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    mid = int(obj["id"])
                    m[mid] = {
                        "title": obj.get("title", f"id_{mid}"),
                        "text": obj.get("text", ""),
                        # keep optional fields if present
                        "year": obj.get("year"),
                        "genres": obj.get("genres"),
                    }
        return m

    # ---------------------- embeddings ----------------------

    def embed_query(self, q: str) -> np.ndarray:
        vec = self.model.encode([q], normalize_embeddings=True)
        return vec.astype("float32")

    # ---------------------- caching helpers ----------------------

    def _cache_key(self, query: str, k: int, page: int, per_page: int) -> str:
        nq = normalize_query(query)
        return f"{self.cache_ns}:search:{self.model_name}:{k}:{page}:{per_page}:{nq}"

    def _cache_get(self, key: str):
        now = time.time()
        # Try Redis
        if self.redis_ok:
            try:
                return self.redis.get(key)
            except RedisError:
                self.redis_ok = False  # fall through to memory

        # Memory fallback
        rec = self._mem.get(key)
        if not rec:
            return None
        exp, blob = rec
        if now > exp:
            self._mem.pop(key, None)
            return None
        return blob

    def _cache_set(self, key: str, payload: bytes, ttl: int = None):
        ttl = ttl or self._mem_ttl
        # Try Redis
        if self.redis_ok:
            try:
                self.redis.setex(key, ttl, payload)
                return
            except RedisError:
                self.redis_ok = False
        # Memory fallback
        self._mem[key] = (time.time() + ttl, payload)

    # ---------------------- reranking ----------------------

    def _hybrid_rerank(
        self,
        query_text: str,
        cand_ids: np.ndarray,
        cand_scores: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """
        Combine semantic scores with TF-IDF cosine on the candidate set:
        hybrid = alpha * semantic + (1 - alpha) * tfidf
        Returns list of (movie_id, hybrid_score) sorted desc.
        """
        docs = [self.id_to_doc.get(int(mid), {}).get("text", "") for mid in cand_ids]

        # If all docs empty, fall back to semantic ranking
        if not any(docs) or all(not (d and d.strip()) for d in docs):
            pairs = list(zip(map(int, cand_ids), cand_scores.astype(float)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs

        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X = tfidf.fit_transform(docs)         # (N, V)
        q_vec = tfidf.transform([query_text]) # (1, V)

        numer = (X @ q_vec.T).toarray().ravel()
        X_norm = np.sqrt((X.power(2)).sum(axis=1)).A.ravel() + 1e-12
        q_norm = float(np.sqrt((q_vec.power(2)).sum()) + 1e-12)
        tfidf_sim = numer / (X_norm * q_norm)

        # normalize semantic to [0,1] before mixing
        sem = cand_scores.astype(float).copy()
        if sem.max() > sem.min():
            sem = (sem - sem.min()) / (sem.max() - sem.min())
        tfidf_sim = np.clip(tfidf_sim, 0.0, 1.0)

        hybrid = self.alpha * sem + (1.0 - self.alpha) * tfidf_sim
        pairs = list(zip(map(int, cand_ids), hybrid))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    # ---------------------- public API ----------------------

    def search(self, query: str, k: int, page: int, per_page: int) -> Tuple[int, List[Dict], bool]:
        """
        Returns (took_ms, results, cached).
        Flow: normalize → cache lookup → semantic@K → hybrid rerank → paginate → cache set.
        """
        key = self._cache_key(query, k, page, per_page)
        cached_blob = self._cache_get(key)
        if cached_blob:
            self.cache_hits += 1
            results = json.loads(cached_blob)
            return 1, results, True

        self.cache_misses += 1
        t0 = time.time()

        # normalized query for retrieval & tfidf
        nq = normalize_query(query)

        # 1) semantic retrieve candidate pool
        qvec = self.embed_query(nq)  # (1, dim)
        sem_scores, sem_ids = self.searcher.search_vectors(qvec, k=self.semantic_k)
        sem_scores = sem_scores[0]
        sem_ids = sem_ids[0]

        # 2) hybrid rerank on candidate set
        ranked = self._hybrid_rerank(nq, sem_ids, sem_scores)

        # 3) paginate within top-k
        start = (page - 1) * per_page
        end = start + per_page
        page_slice = ranked[:k][start:end]

        # 4) format results
        results: List[Dict] = []
        for mid, score in page_slice:
            mid = int(mid)
            meta = self.id_to_doc.get(mid, {"title": f"id_{mid}", "text": ""})
            txt = meta.get("text", "") or ""
            snippet = (txt[:220] + "…") if len(txt) > 220 else txt
            results.append({
                "id": mid,
                "title": meta.get("title", f"id_{mid}"),
                "snippet": snippet,
                "score": float(score),
            })

        took_ms = int((time.time() - t0) * 1000)

        # 5) cache result (non-fatal if Redis down)
        self._cache_set(key, json.dumps(results).encode("utf-8"), ttl=86400)

        return took_ms, results, False

    def stats(self) -> Dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "model": self.model_name,
            "dim": int(self.searcher.dim),
            "doc_count": len(self.searcher.ids),
            "normalized": True,
            "cache_hit_rate": round(hit_rate, 3),
        }
