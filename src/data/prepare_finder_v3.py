"""
STAGE 1 (v3) - token-based chunking. Final version.

History of this file, because the unit errors matter:
  v1: 300 WORDS, single-chunk qrels.
      -> 42% of gold evidence could not fit in one chunk; labels structurally wrong.
  v2a: 512 WORDS, multi-chunk qrels.
      -> word-space sizing ignored that financial text runs ~1.7 tokens/word, so
         99% of chunks blew past the 512-token encoder limit and were silently
         truncated (reranker saw only 64% of each passage).
  v2b: 270 WORDS.
      -> 6% still over the limit, and those are the number-dense TABLE chunks,
         where a single figure explodes into many tokens (worst case 3.7k tokens
         from 270 words). Tables are exactly where financial answers live.
  v3 (this): chunk by TOKENS using the encoder's own tokenizer.
      -> truncation is impossible by construction, tables included.

Config: 448 tokens per chunk, 112 overlap. 448 leaves ~64 tokens of headroom in
the cross-encoder's 512-token budget for the query (measured mean: 20 tokens).

Output: data/finder/processed_v3/
    corpus.jsonl, queries.jsonl, qrels.tsv (graded), answers.jsonl,
    stage1_stats.json

Run:  python src\\data\\prepare_finder_v3.py
"""
from __future__ import annotations
import json, re, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

FINDER = Path("data/finder")
PARQUET = FINDER / "train-00000-of-00001.parquet"
HTML_DIR = FINDER / "10k"
OUT = FINDER / "processed_v3"

TOKENIZER = "BAAI/bge-small-en-v1.5"   # same family as reranker; 512 limit
CHUNK_TOKENS = 448                      # 512 budget - ~64 headroom for the query
CHUNK_OVERLAP = 112                     # 25%
TFIDF_TOPN = 20
COVER_MIN = 0.40                        # relaxed from 0.50: chunks are smaller now
COVER_PRIMARY = 0.70

_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tok


def html_to_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def chunk_by_tokens(text: str, size=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """Split on TOKEN boundaries so no chunk can exceed the encoder limit."""
    ids = tok().encode(text, add_special_tokens=False)
    if not ids:
        return []
    if len(ids) <= size:
        return [text]
    out, start, step = [], 0, max(1, size - overlap)
    while start < len(ids):
        out.append(tok().decode(ids[start:start + size], skip_special_tokens=True))
        start += step
    return out


def words(t: str):
    return set(x for x in re.findall(r"\w+", t.lower()) if len(x) > 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET)

    ev_lens = []
    for _, r in df.iterrows():
        refs = r["references"]
        if refs is None or len(refs) == 0:
            continue
        ev_lens += [len(tok().encode(str(x), add_special_tokens=False)) for x in refs]
    ev_lens = np.array(ev_lens)
    print(f"evidence length (TOKENS): mean {ev_lens.mean():.0f}  "
          f"median {np.median(ev_lens):.0f}  90th {np.percentile(ev_lens,90):.0f}")
    print(f"chunk size: {CHUNK_TOKENS} tokens, overlap {CHUNK_OVERLAP}")
    print(f"  -> evidence fitting one chunk: {100*(ev_lens <= CHUNK_TOKENS).mean():.1f}%")

    files = sorted(glob.glob(str(HTML_DIR / "*.html")))
    print(f"\nparsing {len(files)} filings (token-chunking, slower than word split)...")
    cids, ctexts = [], []
    for n, fp in enumerate(files):
        ticker = Path(fp).stem
        for i, ch in enumerate(chunk_by_tokens(html_to_text(fp))):
            cids.append(f"{ticker}::ch{i}"); ctexts.append(ch)
        if n % 50 == 0:
            print(f"  {n}/{len(files)} filings, {len(cids)} chunks", end="\r")
    print(f"\ncorpus: {len(cids)} chunks")

    # hard verification: nothing may exceed the encoder limit
    samp = np.random.default_rng(0).choice(len(ctexts), size=min(1000, len(ctexts)),
                                           replace=False)
    tl = np.array([len(tok().encode(ctexts[i], add_special_tokens=True)) for i in samp])
    over = int((tl > 512).sum())
    print(f"VERIFY: tokens/chunk mean {tl.mean():.0f} max {tl.max()}  "
          f"over-512: {over}/{len(tl)}")
    assert over == 0, "chunks still exceed 512 tokens - lower CHUNK_TOKENS"

    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as f:
        for cid, ct in zip(cids, ctexts):
            f.write(json.dumps({"_id": cid, "text": ct}) + "\n")
    with open(OUT / "queries.jsonl", "w", encoding="utf-8") as fq, \
         open(OUT / "answers.jsonl", "w", encoding="utf-8") as fa:
        for _, r in df.iterrows():
            qid = str(r["_id"])
            fq.write(json.dumps({"_id": qid, "text": r["text"]}) + "\n")
            fa.write(json.dumps({"_id": qid, "answer": r["answer"],
                                 "category": r["category"], "type": r["type"]}) + "\n")

    print("\nbuilding multi-chunk graded qrels...")
    vec = TfidfVectorizer(stop_words="english", max_features=200_000, ngram_range=(1, 2))
    cmat = vec.fit_transform(ctexts)
    cwords = [words(t) for t in ctexts]

    flat, owner = [], []
    for _, r in df.iterrows():
        refs = r["references"]
        if refs is None or len(refs) == 0:
            continue
        for x in refs:
            flat.append(str(x)); owner.append(str(r["_id"]))
    rmat = vec.transform(flat)

    qrels, B = {}, 128
    for s in range(0, rmat.shape[0], B):
        sims = linear_kernel(rmat[s:s + B], cmat)
        cand = np.argsort(sims, axis=1)[:, ::-1][:, :TFIDF_TOPN]
        for i in range(sims.shape[0]):
            ev_w = words(flat[s + i])
            if not ev_w:
                continue
            qid = owner[s + i]
            for j in cand[i]:
                cov = len(ev_w & cwords[j]) / len(ev_w)
                if cov >= COVER_MIN:
                    g = 2 if cov >= COVER_PRIMARY else 1
                    prev = qrels.setdefault(qid, {}).get(cids[j], 0)
                    qrels[qid][cids[j]] = max(prev, g)
        print(f"  {min(s+B, rmat.shape[0])}/{rmat.shape[0]}", end="\r")
    print()

    with open(OUT / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, d in qrels.items():
            for cid, g in d.items():
                f.write(f"{qid}\t{cid}\t{g}\n")

    per_q = [len(v) for v in qrels.values()]
    stats = {"tokenizer": TOKENIZER, "chunk_tokens": CHUNK_TOKENS,
             "chunk_overlap": CHUNK_OVERLAP, "n_chunks": len(cids),
             "n_questions": int(len(df)), "n_labelled": len(qrels),
             "evidence_tokens_mean": float(ev_lens.mean()),
             "evidence_tokens_median": float(np.median(ev_lens)),
             "evidence_fits_one_chunk_pct": float(100*(ev_lens <= CHUNK_TOKENS).mean()),
             "rel_chunks_per_q_mean": float(np.mean(per_q)),
             "rel_chunks_per_q_median": float(np.median(per_q)),
             "max_tokens_observed": int(tl.max()),
             "cover_min": COVER_MIN, "cover_primary": COVER_PRIMARY}
    json.dump(stats, open(OUT / "stage1_stats.json", "w"), indent=2)

    print(f"\nDONE -> {OUT}")
    print(f"  chunks             : {len(cids)}")
    print(f"  labelled questions : {len(qrels)} ({100*len(qrels)/len(df):.1f}%)")
    print(f"  rel chunks/query   : mean {np.mean(per_q):.2f}  median {np.median(per_q):.0f}")
    print(f"  max tokens observed: {tl.max()} (limit 512)")


if __name__ == "__main__":
    main()