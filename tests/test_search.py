import os

import numpy as np
import pytest

from app.searcher import INDEX_PATH, FaissSearcher


@pytest.mark.skipif(not os.path.exists(INDEX_PATH), reason="index not built yet")
def test_faiss_search_topk_shape():
    s = FaissSearcher()
    # make a fake normalized query vector of right dim
    dim = s.dim
    q = np.random.randn(1, dim).astype("float32")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    scores, ids = s.search_vectors(q, k=3)
    assert scores.shape == (1, 3)
    assert ids.shape == (1, 3)
