"""
STAGE 1 — Prepare FinDER for retrieval experiments.

Input  (you already have these):
    data/finder/train-00000-of-00001.parquet   # 5,703 questions + gold passages + answers
    data/finder/10k/*.html                       # one 10-K per S&P 500 company (by ticker)

Output (this script creates these):
    data/finder/processed/corpus.jsonl    # every searchable chunk: {"_id","text"}
    data/finder/processed/queries.jsonl   # every question:         {"_id","text"}
    data/finder/processed/qrels.tsv       # answer key:  query_id <tab> chunk_id <tab> 1
    data/finder/processed/answers.jsonl   # gold answers (for the generation step later)

Plain English: this turns the raw download into three clean files — the haystack
(corpus), the questions (queries), and the answer key (qrels = which chunk is the
right one for each question). Everything downstream just reads these three files.

Run:  python -m src.data.prepare_finder
"""
from __future__ import annotations
import json, re, glob, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── settings ───────────────────────────────────────────────────────────
FINDER_DIR   = Path("data/finder")
PARQUET      = FINDER_DIR / "train-00000-of-00001.parquet"
HTML_DIR     = FINDER_DIR / "10k"
OUT_DIR      = FINDER_DIR / "processed"
CHUNK_WORDS  = 300          # ~400 tokens; in the 200-500 token target range
CHUNK_OVERLAP= 60           # words of overlap between consecutive chunks
MATCH_THRESH = 0.30         # min TF-IDF cosine to accept a gold-passage -> chunk match


def html_to_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    out, start, step = [], 0, max(1, size - overlap)
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        start += step
    return out


def build_corpus():
    """Parse every 10-K HTML -> chunks. Returns (chunk_ids, chunk_texts)."""
    ids, texts = [], []
    files = sorted(glob.glob(str(HTML_DIR / "*.html")))
    print(f"parsing {len(files)} 10-K files...")
    for fp in files:
        ticker = Path(fp).stem            # e.g. AAPL
        full = html_to_text(fp)
        for i, ch in enumerate(chunk_words(full)):
            ids.append(f"{ticker}::ch{i}")
            texts.append(ch)
    print(f"built corpus of {len(ids)} chunks")
    return ids, texts


def build_qrels(df, chunk_ids, chunk_texts):
    """Match each gold reference passage to its best chunk via TF-IDF top-1.
    Batched: flatten all reference passages, score in blocks against the corpus."""
    print("aligning gold passages to chunks (TF-IDF, batched)...")
    vec = TfidfVectorizer(stop_words="english", max_features=200_000, ngram_range=(1, 2))
    chunk_mat = vec.fit_transform(chunk_texts)

    # flatten every reference passage, remembering which query it came from
    flat_refs, owner_qid = [], []
    for _, row in df.iterrows():
        refs = row["references"]
        if refs is None or len(refs) == 0:
            continue
        for r in refs:
            flat_refs.append(str(r)); owner_qid.append(str(row["_id"]))

    ref_mat = vec.transform(flat_refs)
    qrels, B = {}, 256
    n = ref_mat.shape[0]
    for s in range(0, n, B):
        sims = linear_kernel(ref_mat[s:s + B], chunk_mat)   # (<=B, n_chunks) dense
        best_j = sims.argmax(axis=1)
        best_v = sims.max(axis=1)
        for i in range(sims.shape[0]):
            if best_v[i] >= MATCH_THRESH:
                qid = owner_qid[s + i]
                qrels.setdefault(qid, {})[chunk_ids[best_j[i]]] = 1
        print(f"  matched {min(s + B, n)}/{n} passages", end="\r")

    aligned = len(qrels)
    print(f"\naligned {aligned} queries to gold chunks; "
          f"{len(df) - aligned} dropped (below threshold {MATCH_THRESH})")
    return qrels


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET)

    chunk_ids, chunk_texts = build_corpus()

    # corpus.jsonl
    with open(OUT_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for cid, ct in zip(chunk_ids, chunk_texts):
            f.write(json.dumps({"_id": cid, "text": ct}) + "\n")

    # queries.jsonl + answers.jsonl
    with open(OUT_DIR / "queries.jsonl", "w", encoding="utf-8") as fq, \
         open(OUT_DIR / "answers.jsonl", "w", encoding="utf-8") as fa:
        for _, row in df.iterrows():
            qid = str(row["_id"])
            fq.write(json.dumps({"_id": qid, "text": row["text"]}) + "\n")
            fa.write(json.dumps({"_id": qid, "answer": row["answer"],
                                 "category": row["category"], "type": row["type"]}) + "\n")

    # qrels.tsv
    qrels = build_qrels(df, chunk_ids, chunk_texts)
    with open(OUT_DIR / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, docs in qrels.items():
            for cid in docs:
                f.write(f"{qid}\t{cid}\t1\n")

    print("\nDONE. wrote to", OUT_DIR)
    print(f"  corpus chunks : {len(chunk_ids)}")
    print(f"  queries       : {len(df)}")
    print(f"  scored queries: {len(qrels)}")


if __name__ == "__main__":
    main()