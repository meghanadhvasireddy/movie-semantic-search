import json, os, time
from typing import Dict, Tuple, List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.searcher import FaissSearcher, META_PATH

DOCS_PATH = "data/processed/docs.jsonl"

class SearchService:
    def __init__(self):
        # load meta to get the exact model used for corpus embeddings
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model_name = self.meta["model"]
        self.model = SentenceTransformer(self.model_name)  # load ONCE
        self.searcher = FaissSearcher()                    # loads faiss index lazily
        self.id_to_doc = self._load_docs_map()            # for titles/snippets

    def _load_docs_map(self) -> Dict[int, Dict]:
        """
        Build a lightweight lookup: id -> {title, text}
        Used to construct human-friendly results. If your dataset is large
        and RAM is tight, you can store only title and first N chars of text.
        """
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
        # IMPORTANT: normalize to match corpus embeddings
        vec = self.model.encode([q], normalize_embeddings=True)
        return vec.astype("float32")

    def search(self, query: str, k: int) -> Tuple[int, List[Dict]]:
        """
        Returns (took_ms, results_list)
        results_list items: {id, title, snippet, score}
        """
        t0 = time.time()
        qvec = self.embed_query(query)
        scores, ids = self.searcher.search_vectors(qvec, k=k)
        took_ms = int((time.time() - t0) * 1000)

        out: List[Dict] = []
        for score, mid in zip(scores[0], ids[0]):
            mid = int(mid)
            meta = self.id_to_doc.get(mid, {"title": f"id_{mid}", "text": ""})
            text = meta["text"] or ""
            # simple snippet: first 220 chars
            snippet = (text[:220] + "…") if len(text) > 220 else text
            out.append({
                "id": mid,
                "title": meta["title"],
                "snippet": snippet,
                "score": float(score)
            })
        return took_ms, out

    def stats(self) -> Dict:
        return {
            "model": self.model_name,
            "dim":  int(self.searcher.dim),
            "doc_count": len(self.searcher.ids),
            "normalized": True,
        }
