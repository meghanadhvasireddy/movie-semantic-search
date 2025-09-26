import json
import os
from typing import Tuple

import faiss
import numpy as np

ART_DIR = "artifacts"
EMB_PATH = os.path.join(ART_DIR, "embeddings.npy")
IDS_PATH = os.path.join(ART_DIR, "ids.npy")
META_PATH = os.path.join(ART_DIR, "meta.json")
INDEX_PATH = os.path.join(ART_DIR, "faiss.index")


class FaissSearcher:
    """
    Loads embeddings + ids + FAISS index and performs top-K similarity search.
    Assumes embeddings are L2-normalized; uses inner product (cosine-equivalent).
    """

    def __init__(self):
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Missing {META_PATH}. Did you run Day 3 indexer?")
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.dim = int(self.meta["dim"])

        # lazy-load arrays only when needed
        self._ids = None
        self._index = None

    @property
    def ids(self) -> np.ndarray:
        if self._ids is None:
            self._ids = np.load(IDS_PATH)
        return self._ids

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            if os.path.exists(INDEX_PATH):
                self._index = faiss.read_index(INDEX_PATH)
            else:
                raise FileNotFoundError(f"Missing {INDEX_PATH}. Build it first.")
        return self._index

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        # ensure query is unit length; guards against accidental non-normalized input
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / norms

    def search_vectors(
        self, query_vectors: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_vectors: shape (Q, dim)
        returns (scores, id_array) where:
          scores: (Q, k) similarity scores (inner product)
          id_array: (Q, k) dataset integer ids (your movie ids)
        """
        if query_vectors.shape[1] != self.dim:
            raise ValueError(f"query dim {query_vectors.shape[1]} != index dim {self.dim}")
        q = self._normalize(query_vectors.astype("float32"))
        scores, idx = self.index.search(q, k)  # idx are ROW NUMBERS into embeddings
        data_ids = self.ids[idx]  # map row numbers to your movie 'id'
        return scores, data_ids
