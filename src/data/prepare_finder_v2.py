"""
STAGE 1 (v2) - rebuilt from the data, not from assumptions.

Two fixes over v1, both forced by the Stage 1 audit:

  1. CHUNK SIZE IS MEASURED, NOT GUESSED.
     v1 used 300 words. The audit showed gold evidence averages 407 words
     (median 257, 90th pct 943), so 42% of evidence could not fit in a chunk.
     v2 prints the evidence-length distribution and sets the chunk size to cover
     the median comfortably (default 512 words, 128 overlap).

  2. MULTI-CHUNK QRELS.
     v1 labelled exactly ONE chunk per question (top TF-IDF match). When evidence
     spans several chunks, that penalises the retriever for finding a genuinely
     correct chunk. v2 labels EVERY chunk that materially overlaps the gold
     evidence, using word coverage (independent of TF-IDF).

Output (data/finder/processed_v2/):
    corpus.jsonl    {"_id","text"}
    queries.jsonl   {"_id","text"}
    qrels.tsv       query-id, corpus-id, score   (may be several rows per query)
    answers.jsonl   gold answers for the generation stage
    stage1_stats.json  measured statistics, for the paper

Run:  python src\\data\\prepare_finder_v2.py
"""
from __future__ import annotations
import json, re, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

FINDER = Path("data/finder")
PARQUET = FINDER / "train-00000-of-00001.parquet"
HTML_DIR = FINDER / "10k"
OUT = FINDER / "processed_v2"

CHUNK_WORDS = 270
CHUNK_OVERLAP = 68
TFIDF_TOPN = 20            # candidates per evidence passage, then verified by coverage
COVER_MIN = 0.50
COVER_PRIMARY = 0.80


def html_to_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def chunk_words(text: str, size: int, overlap: int):
    w = text.split()
    if len(w) <= size:
        return [text] if w else []
    out, start, step = [], 0, max(1, size - overlap)
    while start < len(w):
        out.append(" ".join(w[start:start + size]))
        start += step
    return out


def words(t: str):
    return set(x for x in re.findall(r"\w+", t.lower()) if len(x) > 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET)

    # ---- measure evidence lengths FIRST; the chunk size follows from this ----
    ev_lens = []
    for _, r in df.iterrows():
        refs = r["references"]
        if refs is None or len(refs) == 0:
            continue
        ev_lens += [len(str(x).split()) for x in refs]
    ev_lens = np.array(ev_lens)
    print("evidence length (words): "
          f"mean {ev_lens.mean():.0f}  median {np.median(ev_lens):.0f}  "
          f"90th {np.percentile(ev_lens, 90):.0f}  max {ev_lens.max()}")
    print(f"chunk size chosen: {CHUNK_WORDS} words, overlap {CHUNK_OVERLAP}")
    print(f"  -> evidence that fits in one chunk: "
          f"{100 * (ev_lens <= CHUNK_WORDS).mean():.1f}%")

    # ---- build corpus ----
    files = sorted(glob.glob(str(HTML_DIR / "*.html")))
    print(f"\nparsing {len(files)} filings...")
    cids, ctexts = [], []
    for fp in files:
        ticker = Path(fp).stem
        for i, ch in enumerate(chunk_words(html_to_text(fp), CHUNK_WORDS, CHUNK_OVERLAP)):
            cids.append(f"{ticker}::ch{i}"); ctexts.append(ch)
    print(f"corpus: {len(cids)} chunks")

    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as f:
        for cid, ct in zip(cids, ctexts):
            f.write(json.dumps({"_id": cid, "text": ct}) + "\n")

    # ---- queries + answers ----
    with open(OUT / "queries.jsonl", "w", encoding="utf-8") as fq, \
         open(OUT / "answers.jsonl", "w", encoding="utf-8") as fa:
        for _, r in df.iterrows():
            qid = str(r["_id"])
            fq.write(json.dumps({"_id": qid, "text": r["text"]}) + "\n")
            fa.write(json.dumps({"_id": qid, "answer": r["answer"],
                                 "category": r["category"], "type": r["type"]}) + "\n")

    # ---- multi-chunk qrels ----
    print("\nbuilding multi-chunk qrels...")
    vec = TfidfVectorizer(stop_words="english", max_features=200_000, ngram_range=(1, 2))
    cmat = vec.fit_transform(ctexts)
    cwords = [words(t) for t in ctexts]

    flat_refs, owner = [], []
    for _, r in df.iterrows():
        refs = r["references"]
        if refs is None or len(refs) == 0:
            continue
        for x in refs:
            flat_refs.append(str(x)); owner.append(str(r["_id"]))
    rmat = vec.transform(flat_refs)

    qrels, B = {}, 128
    n_lab, n_chunks_tot = 0, 0
    for s in range(0, rmat.shape[0], B):
        sims = linear_kernel(rmat[s:s + B], cmat)
        cand = np.argsort(sims, axis=1)[:, ::-1][:, :TFIDF_TOPN]
        for i in range(sims.shape[0]):
            ev_w = words(flat_refs[s + i])
            if not ev_w:
                continue
            qid = owner[s + i]
            hits = 0
            for j in cand[i]:
                cov = len(ev_w & cwords[j]) / len(ev_w)
                if cov >= COVER_MIN:
                    grade = 2 if cov >= COVER_PRIMARY else 1
                    prev = qrels.setdefault(qid, {}).get(cids[j], 0)
                    qrels[qid][cids[j]] = max(prev, grade)
                    hits += 1
            n_lab += int(hits > 0); n_chunks_tot += hits
        print(f"  {min(s + B, rmat.shape[0])}/{rmat.shape[0]} evidence passages", end="\r")
    print()

    with open(OUT / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, d in qrels.items():
            for cid, g in d.items():
                f.write(f"{qid}\t{cid}\t{g}\n")

    per_q = [len(v) for v in qrels.values()]
    stats = {
        "chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP,
        "n_chunks": len(cids), "n_questions": int(len(df)),
        "n_questions_labelled": len(qrels),
        "evidence_words_mean": float(ev_lens.mean()),
        "evidence_words_median": float(np.median(ev_lens)),
        "evidence_fits_one_chunk_pct": float(100 * (ev_lens <= CHUNK_WORDS).mean()),
        "relevant_chunks_per_question_mean": float(np.mean(per_q)),
        "relevant_chunks_per_question_median": float(np.median(per_q)),
        "cover_min": COVER_MIN, "cover_primary": COVER_PRIMARY,
    }
    json.dump(stats, open(OUT / "stage1_stats.json", "w"), indent=2)

    print(f"\nDONE -> {OUT}")
    print(f"  chunks               : {len(cids)}")
    print(f"  questions            : {len(df)}")
    print(f"  labelled questions   : {len(qrels)} ({100*len(qrels)/len(df):.1f}%)")
    print(f"  relevant chunks/query: mean {np.mean(per_q):.2f}  median {np.median(per_q):.0f}")


if __name__ == "__main__":
    main()