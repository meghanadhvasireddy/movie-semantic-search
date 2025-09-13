import json, os, re, time
from typing import Dict, Tuple, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from app.searcher import FaissSearcher, META_PATH
import redis

DOCS_PATH = "data/processed/docs.jsonl"

def normalize_query(q: str) -> str:
    # lower, remove punctuation except spaces, collapse whitespace
    q = q.lower()
    q = re.sub(r"[^\w\s]", " ", q)      # remove punctuation
    q = re.sub(r"\s+", " ", q).strip()  # collapse spaces
    return q

class SearchService:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        # meta/model
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model_name = self.meta["model"]
        self.model = SentenceTransformer(self.model_name)

        # index + docs
        self.searcher = FaissSearcher()
        self.id_to_doc = self._load_docs_map()

        # cache
        self.redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self.cache_hits = 0
        self.cache_misses = 0

        # knobs
        self.semantic_k = 50    # how many to fetch from FAISS before rerank
        self.alpha = 0.8        # weight for semantic score in hybrid

    def _load_docs_map(self) -> Dict[int, Dict]:
        m: Dict[int, Dict] = {}
        if os.path.exists(DOCS_PATH):
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    mid = int(obj["id"])
                    m[mid] = {
                        "title": obj.get("title", f"id_{mid}"),
                        "text":  obj.get("text", "")
                    }
        return m

    def embed_query(self, q: str) -> np.ndarray:
        vec = self.model.encode([q], normalize_embeddings=True)
        return vec.astype("float32")

    def _cache_key(self, query: str, k: int, page: int, per_page: int) -> str:
        # IMPORTANT: use normalized query for cache key so punctuation/case don’t split cache
        nq = normalize_query(query)
        return f"search:{self.model_name}:{k}:{page}:{per_page}:{nq}"

    def _hybrid_rerank(
        self,
        query_text: str,
        cand_ids: np.ndarray,
        cand_scores: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Given candidate ids and their semantic scores, compute a TF-IDF score
        on those candidates and combine: alpha*semantic + (1-alpha)*tfidf.
        Returns list of (movie_id, hybrid_score) sorted desc.
        """
        docs = [self.id_to_doc.get(int(mid), {}).get("text", "") for mid in cand_ids]
        # TF-IDF on the candidate set only (small and fast)
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X = tfidf.fit_transform(docs)            # shape: (N_docs, V)
        q_vec = tfidf.transform([query_text])    # shape: (1, V)
        # cosine sim for tfidf space
        # (X * q.T) / (||X||*||q||)  → but sklearn vectors are L2-normalized by default when using TfidfVectorizer? (not exactly),
        # so compute manually:
        numer = (X @ q_vec.T).toarray().ravel()  # dot products
        X_norm = np.sqrt((X.power(2)).sum(axis=1)).A.ravel() + 1e-12
        q_norm = np.sqrt((q_vec.power(2)).sum()) + 1e-12
        tfidf_sim = numer / (X_norm * q_norm)

        # normalize both scores to [0,1] to mix fairly
        sem = cand_scores.copy()
        if sem.max() > sem.min():
            sem = (sem - sem.min()) / (sem.max() - sem.min())
        # tfidf already roughly in [0,1] (cosine), but clamp just in case
        tfidf_sim = np.clip(tfidf_sim, 0.0, 1.0)

        hybrid = self.alpha * sem + (1.0 - self.alpha) * tfidf_sim
        pairs = list(zip(map(int, cand_ids), hybrid))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def search(self, query: str, k: int, page: int, per_page: int) -> Tuple[int, List[Dict], bool]:
        """
        Returns (took_ms, results, cached)
        Applies: normalize → cache lookup → semantic@K → TF-IDF rerank → paginate.
        """
        key = self._cache_key(query, k, page, per_page)
        cached_blob = self.redis.get(key)
        if cached_blob:
            self.cache_hits += 1
            results = json.loads(cached_blob)
            return 1, results, True

        self.cache_misses += 1
        t0 = time.time()

        # normalized query for retrieval & tfidf
        nq = normalize_query(query)
        # 1) semantic retrieve top-N
        qvec = self.embed_query(nq)
        sem_scores, sem_ids = self.searcher.search_vectors(qvec, k=self.semantic_k)
        sem_scores = sem_scores[0]
        sem_ids = sem_ids[0]

        # 2) hybrid rerank
        ranked = self._hybrid_rerank(nq, sem_ids, sem_scores)

        # 3) paginate
        start = (page - 1) * per_page
        end = start + per_page
        page_slice = ranked[:k]  # ensure we don't exceed requested k overall
        page_slice = page_slice[start:end]

        # 4) format
        results: List[Dict] = []
        for mid, score in page_slice:
            meta = self.id_to_doc.get(mid, {"title": f"id_{mid}", "text": ""})
            txt = meta["text"] or ""
            snippet = (txt[:220] + "…") if len(txt) > 220 else txt
            results.append({
                "id": int(mid),
                "title": meta["title"],
                "snippet": snippet,
                "score": float(score)
            })

        took_ms = int((time.time() - t0) * 1000)
        # 5) cache result for 24h
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
