import json
import os

import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
PLOT_FILE = os.path.join(RAW_DIR, "plot_summaries.txt")
META_FILE = os.path.join(RAW_DIR, "movie.metadata.tsv")
OUT_JSONL = os.path.join(OUT_DIR, "docs.jsonl")
OUT_PARQUET = os.path.join(OUT_DIR, "docs.parquet")  # optional fast reload

os.makedirs(OUT_DIR, exist_ok=True)


def clean_text(s: str) -> str:
    """Make text tidy: remove newlines, extra spaces, leading/trailing blanks."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = " ".join(s.split())  # collapse multiple spaces
    return s


def get_year(date_str: str):
    """Try to extract a 4-digit year from the date column."""
    if not isinstance(date_str, str) or len(date_str) < 4:
        return None
    y = date_str[:4]
    return int(y) if y.isdigit() else None


def ingest_cmu(plot_path: str, meta_path: str) -> pd.DataFrame:
    # 1) Load plot summaries: <wiki_id>\t<plot>
    #    Keep as strings; some plots contain punctuation/commas.
    plots = pd.read_csv(
        plot_path,
        sep="\t",
        header=None,
        names=["wiki_id", "plot"],
        dtype={0: str, 1: str},
        quoting=3,  # QUOTE_NONE: don't treat quotes specially
        encoding="utf-8",  # try utf-8 first; switch to latin-1 if needed
        on_bad_lines="skip",
    )
    # Basic clean
    plots["plot"] = plots["plot"].fillna("").map(clean_text)
    plots = plots[plots["plot"].str.len() > 0]

    # 2) Load metadata: we only need first 4 columns
    meta = pd.read_csv(
        meta_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["wiki_id", "freebase_id", "title", "date"],
        dtype=str,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    meta["title"] = meta["title"].fillna("").map(clean_text)
    meta = meta[meta["title"].str.len() > 0]
    meta["year"] = meta["date"].map(get_year)

    # 3) Join on wiki_id (the glue key)
    df = plots.merge(meta[["wiki_id", "title", "year"]], on="wiki_id", how="left")

    # 4) Final clean + select columns
    df = df.rename(columns={"plot": "text"})
    df["text"] = df["text"].fillna("").map(clean_text)
    df["title"] = df["title"].fillna("").map(clean_text)
    df = df[(df["title"].str.len() > 0) & (df["text"].str.len() > 0)]
    # Drop perfect duplicates (sometimes the corpus has repeats)
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    # 5) Add numeric id column (0..N-1)
    df.insert(0, "id", df.index.astype(int))

    # Keep core fields first; year is optional
    cols = ["id", "title", "text", "year"] if "year" in df.columns else ["id", "title", "text"]
    return df[cols]


def save_jsonl(df: pd.DataFrame, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {
                k: (None if (pd.isna(row[k]) if k in row else False) else row[k])
                for k in df.columns
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(df)} docs → {out_path}")


def maybe_save_parquet(df: pd.DataFrame, out_path: str):
    try:
        df.to_parquet(out_path, index=False)
        print(f"[OK] Parquet copy → {out_path}")
    except Exception as e:
        print(f"[WARN] Could not save parquet: {e}")


if __name__ == "__main__":
    # Sanity checks
    if not os.path.exists(PLOT_FILE):
        raise SystemExit(f"Missing {PLOT_FILE}. Place plot_summaries.txt in data/raw/")
    if not os.path.exists(META_FILE):
        raise SystemExit(f"Missing {META_FILE}. Place movie.metadata.tsv in data/raw/")

    print("[*] Ingesting CMU Movie Summary Corpus…")
    df = ingest_cmu(PLOT_FILE, META_FILE)

    # Optional quick sanity preview
    print(df.head(3).to_string(index=False))

    # Save outputs
    save_jsonl(df, OUT_JSONL)
    maybe_save_parquet(df, OUT_PARQUET)

    # Optional: also save a small sample for tests/benchmarks
    sample_path = os.path.join(OUT_DIR, "sample_500.jsonl")
    df.sample(min(500, len(df)), random_state=42).to_json(
        sample_path, orient="records", lines=True, force_ascii=False
    )
    print(f"[OK] Sample for tests → {sample_path}")
