import argparse
import json
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.searcher import (
    ART_DIR, EMB_PATH, IDS_PATH, META_PATH, INDEX_PATH, FaissSearcher
)

DOCS_PATH = "data/processed/docs.jsonl"  # used only to print titles/snippets for demo

def build_index(index_type: str = "flatip") -> None:
    """
    Build a FAISS index from embeddings and write to disk.
    index_type: 'flatip' (exact, inner product) or 'hnsw' (approximate, faster)
    """
    if not os.path.exists(EMB_PATH) or not os.path.exists(IDS_PATH) or not os.path.exists(META_PATH):
        raise SystemExit("Missing artifacts. Run Day 3 indexer first.")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    dim = int(meta["dim"])

    embs = np.load(EMB_PATH).astype("float32")  # shape (N, dim), already normalized
    print(f"[build] embeddings shape: {embs.shape}; dim={dim}")

    if index_type.lower() == "flatip":
        index = faiss.IndexFlatIP(dim)
    elif index_type.lower() == "hnsw":
        # HNSW for inner product: create HNSW with M=32; you can tune efSearch later
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 64
    else:
        raise ValueError("index_type must be 'flatip' or 'hnsw'")

    index.add(embs)  # add all vectors
    faiss.write_index(index, INDEX_PATH)
    print(f"[build] wrote index to {INDEX_PATH}")

def _load_titles() -> dict:
    """
    Map movie id -> title (to print friendly results in CLI demo).
    """
    id_to_title = {}
    if not os.path.exists(DOCS_PATH):
        return id_to_title
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            id_to_title[int(obj["id"])] = obj.get("title", f"id_{obj['id']}")
    return id_to_title

def embed_query(model_name: str, text: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vec = model.encode([text], normalize_embeddings=True)
    return vec.astype("float32")

def cli_search(query: str, k: int = 5):
    # read model name from meta to ensure the same embedder is used for queries
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_name = meta["model"]

    # embed query the same way corpus was embedded
    qvec = embed_query(model_name, query)

    # search
    searcher = FaissSearcher()
    scores, ids = searcher.search_vectors(qvec, k=k)

    # pretty print
    id_to_title = _load_titles()
    print(f'\n[results] query: "{query}"\n')
    for rank, (score, mid) in enumerate(zip(scores[0], ids[0]), 1):
        title = id_to_title.get(int(mid), f"id_{int(mid)}")
        print(f"{rank:2d}. {title:40s}  score={score:.3f}  (id={int(mid)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="Build and save FAISS index")
    b.add_argument("--type", default="flatip", choices=["flatip", "hnsw"])

    s = sub.add_parser("search", help="Search the index with a text query")
    s.add_argument("--query", required=True)
    s.add_argument("--k", type=int, default=5)

    args = p.parse_args()

    if args.cmd == "build":
        build_index(index_type=args.type)
    elif args.cmd == "search":
        cli_search(args.query, k=args.k)
    else:
        p.print_help()
