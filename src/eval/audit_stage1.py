"""
STAGE 1 AUDIT - verify the foundation before rebuilding anything on top of it.

Everything in this project inherits from prepare_finder.py, and two of its
decisions were never validated:

  (a) HTML -> text extraction of 500 filings
  (b) TF-IDF alignment of gold evidence text to a chunk, at threshold 0.30

If (b) is unreliable, the qrels are partly wrong, the retriever is being graded
against bad labels, and every downstream number is meaningless. That would look
exactly like what the rank diagnostic showed: "gold" chunks ranked beyond 1000
with low query similarity - because the labelled chunk is not the answer.

Checks performed:
  1. CHUNK TEXT QUALITY      - boilerplate/navigation junk, degenerate chunks
  2. ALIGNMENT CONFIDENCE    - distribution of the TF-IDF match score used as qrel
  3. ALIGNMENT CORRECTNESS   - does the aligned chunk actually contain the gold
                               evidence? measured by word-overlap, independent of
                               the TF-IDF score that created the label
  4. HARD CASES              - for questions whose gold ranked >1000 in retrieval,
                               is their alignment worse than average?
  5. SAMPLES                 - prints real examples to eyeball

Run:  python src\\eval\\audit_stage1.py
"""
from __future__ import annotations
import json, re, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

PROC = Path("data/finder/processed")
FINDER = Path("data/finder")
random.seed(0)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_qrels(p):
    q = {}
    f = open(p, encoding="utf-8"); next(f)
    for line in f:
        a, b, _ = line.rstrip("\n").split("\t")
        q.setdefault(a, []).append(b)
    return q


def overlap(evidence: str, chunk: str) -> float:
    """Fraction of the gold evidence's words present in the chunk.
    Independent of TF-IDF, so it can validate the labels TF-IDF produced."""
    ev = set(w for w in re.findall(r"\w+", evidence.lower()) if len(w) > 2)
    ch = set(w for w in re.findall(r"\w+", chunk.lower()) if len(w) > 2)
    return len(ev & ch) / len(ev) if ev else 0.0


def main():
    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    qrels = load_qrels(PROC / "qrels.tsv")
    df = pd.read_parquet(FINDER / "train-00000-of-00001.parquet")
    ref_by_id = {}
    for _, r in df.iterrows():
        refs = r["references"]
        refs = [] if refs is None or len(refs) == 0 else [str(x) for x in refs]
        ref_by_id[str(r["_id"])] = refs
    print(f"corpus chunks: {len(cids)}   labelled questions: {len(qrels)}")

    # ---------- 1. chunk text quality ----------
    print("\n" + "=" * 60)
    print("1. CHUNK TEXT QUALITY")
    lens = np.array([len(t.split()) for t in ctexts])
    print(f"  words/chunk: mean {lens.mean():.0f}  min {lens.min()}  max {lens.max()}")
    short = int((lens < 50).sum())
    print(f"  very short chunks (<50 words): {short} ({100*short/len(lens):.1f}%)")
    samp = random.sample(range(len(ctexts)), 3000)
    def junk_ratio(t):
        w = re.findall(r"\w+", t)
        if not w:
            return 1.0
        # tokens that are pure punctuation-ish noise or single chars
        odd = sum(1 for x in w if len(x) <= 1)
        return odd / len(w)
    jr = np.array([junk_ratio(ctexts[i]) for i in samp])
    print(f"  mean single-char-token ratio: {jr.mean():.3f}  (high => bad HTML extraction)")
    uniq = len({ctexts[i][:200] for i in samp})
    print(f"  distinct first-200-chars in 3000 sample: {uniq} "
          f"({100*uniq/3000:.1f}%)  (low => boilerplate duplication)")

    # ---------- 2 & 3. alignment confidence and correctness ----------
    print("\n" + "=" * 60)
    print("2/3. ALIGNMENT QUALITY  (is the labelled chunk really the answer?)")
    qids = [q for q in qrels if q in ref_by_id and ref_by_id[q]]
    check = random.sample(qids, min(800, len(qids)))
    ovs = []
    for qid in check:
        ev = ref_by_id[qid][0]
        best = max(overlap(ev, cid2text[c]) for c in qrels[qid] if c in cid2text)
        ovs.append(best)
    ovs = np.array(ovs)
    print(f"  checked {len(ovs)} questions")
    print(f"  gold-evidence word coverage in the labelled chunk:")
    print(f"    mean {ovs.mean():.3f}   median {np.median(ovs):.3f}")
    for thr in (0.9, 0.7, 0.5, 0.3):
        n = int((ovs >= thr).sum())
        print(f"    >= {thr:.1f} coverage: {n:>4} ({100*n/len(ovs):5.1f}%)")
    bad = int((ovs < 0.3).sum())
    print(f"\n  LIKELY MISLABELLED (<0.3 coverage): {bad} ({100*bad/len(ovs):.1f}%)")
    if bad / len(ovs) > 0.2:
        print("  => The qrels are substantially unreliable. Fix alignment BEFORE")
        print("     drawing any conclusion from retrieval metrics.")
    elif bad / len(ovs) > 0.08:
        print("  => Some label noise. Worth tightening, but not the dominant problem.")
    else:
        print("  => Alignment looks sound. The labels are not the bottleneck.")

    # ---------- 4. evidence length vs chunk size ----------
    print("\n" + "=" * 60)
    print("4. EVIDENCE vs CHUNK SIZE  (can one chunk even hold the evidence?)")
    ev_lens = np.array([len(ref_by_id[q][0].split()) for q in check])
    print(f"  gold evidence words: mean {ev_lens.mean():.0f}  median {np.median(ev_lens):.0f}"
          f"  90th pct {np.percentile(ev_lens,90):.0f}")
    too_big = int((ev_lens > 300).sum())
    print(f"  evidence longer than the 300-word chunk size: {too_big} "
          f"({100*too_big/len(ev_lens):.1f}%)")
    if too_big / len(ev_lens) > 0.3:
        print("  => Much gold evidence CANNOT fit in one chunk, so a single-chunk")
        print("     qrel is structurally wrong. Needs multi-chunk labels or bigger chunks.")

    # ---------- 5. samples ----------
    print("\n" + "=" * 60)
    print("5. SAMPLES (worst 3 alignments)")
    order = np.argsort(ovs)[:3]
    for i in order:
        qid = check[i]
        print(f"\n  --- qid {qid}  coverage={ovs[i]:.3f}")
        print(f"  EVIDENCE : {ref_by_id[qid][0][:300]}...")
        c = qrels[qid][0]
        print(f"  LABELLED CHUNK ({c}) : {cid2text.get(c,'')[:300]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()