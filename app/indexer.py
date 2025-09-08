import os
import json
import math
from typing import Iterator, List, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/processed/docs.jsonl"       # full dataset
SAMPLE_PATH = "data/processed/sample_500.jsonl"  # small sanity set
ART_DIR = "artifacts"
META_PATH = os.path.join(ART_DIR, "meta.json")
EMB_PATH = os.path.join(ART_DIR, "embeddings.npy")
IDS_PATH = os.path.join(ART_DIR, "ids.npy")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, good

os.makedirs(ART_DIR, exist_ok=True)

def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_documents(path: str) -> Tuple[List[int], List[str]]:
    ids, texts = [], []
    for row in read_jsonl(path):
        # Expect row has id, title, text (and maybe year)
        if "id" in row and "text" in row and isinstance(row["text"], str):
            t = row["text"].strip()
            if t:
                ids.append(int(row["id"]))
                texts.append(t)
    return ids, texts

def chunk(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def embed_corpus(model_name: str, texts: List[str], batch_size: int = 256) -> np.ndarray:
    """
    Returns a 2D numpy array of shape (N, D) where D is embedding dim.
    """
    model = SentenceTransformer(model_name)
    embeddings = []
    for batch in tqdm(chunk(texts, batch_size), total=math.ceil(len(texts)/batch_size), desc="Embedding"):
        # normalize_embeddings=True makes vectors length ~1 (good for cosine/IP)
        vecs = model.encode(batch, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(np.asarray(vecs, dtype=np.float32))
    return np.vstack(embeddings)

def save_artifacts(embeddings: np.ndarray, ids: List[int], meta: dict):
    np.save(EMB_PATH, embeddings)
    np.save(IDS_PATH, np.asarray(ids, dtype=np.int64))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved: {EMB_PATH}, {IDS_PATH}, {META_PATH}")

def estimate_memory(num_docs: int, dim: int = 384, dtype_size: int = 4) -> float:
    """
    Rough memory in MB for embeddings array (ignores overhead).
    num_docs * dim * 4 bytes (float32)
    """
    bytes_needed = num_docs * dim * dtype_size
    return bytes_needed / (1024 * 1024)

def main(use_sample=False, batch_size=256):
    path = SAMPLE_PATH if use_sample else DOCS_PATH
    ids, texts = load_documents(path)

    if len(ids) == 0:
        raise SystemExit(f"No documents found in {path}. Did Day 2 produce docs.jsonl?")

    # quick estimate: how big will the embeddings array be?
    est_mb = estimate_memory(len(ids), dim=384)
    print(f"Docs: {len(ids)} | Estimated embedding array size: ~{est_mb:.1f} MB")

    embs = embed_corpus(MODEL_NAME, texts, batch_size=batch_size)
    assert embs.shape[0] == len(ids), "Row count mismatch"
    dim = embs.shape[1]
    print(f"Embeddings shape: {embs.shape}  (N={embs.shape[0]}, dim={dim})")

    meta = {
        "model": MODEL_NAME,
        "dim": int(dim),
        "doc_count": int(len(ids)),
        "normalized": True,
        "notes": "Vectors are L2-normalized for cosine/IP search."
    }
    save_artifacts(embs, ids, meta)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="Embed only sample_500.jsonl for a quick sanity run")
    p.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    args = p.parse_args()

    main(use_sample=args.sample, batch_size=args.batch_size)
