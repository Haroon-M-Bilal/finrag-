"""
STAGE 1 (v4) - token-based chunking + filing-restricted union-coverage qrels.

History of this file, because the errors matter:
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
  v3: chunk by TOKENS using the encoder's own tokenizer.
      -> truncation impossible by construction. But the LABELLER was still wrong:
         references were matched against all ~182k chunks globally, so 10-K
         boilerplate collided across companies. 52.8% of questions ended up with
         gold evidence in more than one filing (max 38). Under a filing-level
         train/test split that is a leakage path, not just noise.
  v4 (this): three labelling fixes.
      1. Gold chunks restricted to the question's OWN filing (resolved by summed
         TF-IDF similarity over its references, with an ambiguity flag).
      2. UNION coverage across selected chunks, not per-chunk fraction. Per-chunk
         coverage falls automatically as chunks shrink, which is the only reason
         COVER_MIN had to move 0.50 -> 0.40 between v2 and v3. Union coverage is a
         property of the evidence, so the threshold is stable across chunk sizes
         and v2/v3/v4 are comparable.
      3. Overlap counts MULTIPLICITY (Counter, not set), so a chunk mentioning
         "total"/"revenues"/"2023" once no longer fully matches a reference that
         uses them twenty times.

Config: 448 tokens per chunk, 112 overlap. 448 leaves ~64 tokens of headroom in
the cross-encoder's 512-token budget for the query (measured mean: 20 tokens).

Output: data/finder/processed_v4/
    corpus.jsonl, queries.jsonl, qrels.tsv (graded), answers.jsonl,
    question_filing.json, stage1_stats.json

Run:  python -u src\\data\\prepare_finder_v4.py
"""
from __future__ import annotations

import json, re, glob, warnings
from collections import Counter
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
OUT = FINDER / "processed_v4"

TOKENIZER = "BAAI/bge-small-en-v1.5"   # same family as reranker; 512 limit
CHUNK_TOKENS = 448                      # 512 budget - ~64 headroom for the query
CHUNK_OVERLAP = 112                     # 25%

COVER_TARGET = 0.80    # union of selected chunks must cover this much of the reference
GRADE2_SHARE = 0.50    # a chunk contributing >= this share of the reference is grade 2
FILING_TOPN = 40       # candidate chunks considered within the resolved filing
RESOLVE_TOPN = 200     # global candidates used only to resolve question -> filing
AMBIG_MARGIN = 1.25    # top filing must beat runner-up by this ratio, else flagged

TOKRE = re.compile(r"[a-z0-9]+")

_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tok


def wcount(t: str) -> Counter:
    return Counter(TOKRE.findall(t.lower()))


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET)

    # ---------------------------------------------------------- evidence length report
    ev_lens = []
    for _, r in df.iterrows():
        refs = r["references"]
        if refs is None or len(refs) == 0:
            continue
        ev_lens += [len(tok().encode(str(x), add_special_tokens=False)) for x in refs]
    ev_lens = np.array(ev_lens)
    print(f"evidence length (TOKENS): mean {ev_lens.mean():.0f}  "
          f"median {np.median(ev_lens):.0f}  90th {np.percentile(ev_lens, 90):.0f}")
    print(f"chunk size: {CHUNK_TOKENS} tokens, overlap {CHUNK_OVERLAP}")
    print(f"  -> evidence fitting one chunk: {100 * (ev_lens <= CHUNK_TOKENS).mean():.1f}%")

    # ------------------------------------------------------------------- build corpus
    files = sorted(glob.glob(str(HTML_DIR / "*.html")))
    print(f"\nparsing {len(files)} filings (token-chunking, slower than word split)...")

    tickers = [Path(fp).stem for fp in files]
    assert len(set(tickers)) == len(tickers), "duplicate ticker - chunk IDs would collide"

    cids, ctexts = [], []
    for n, fp in enumerate(files):
        ticker = Path(fp).stem
        for i, ch in enumerate(chunk_by_tokens(html_to_text(fp))):
            cids.append(f"{ticker}::ch{i}")
            ctexts.append(ch)
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

    # ------------------------------------------------------------------ TF-IDF index
    print("\nbuilding TF-IDF index...")
    vec = TfidfVectorizer(stop_words="english", max_features=200_000, ngram_range=(1, 2))
    cmat = vec.fit_transform(ctexts)

    tick_of = [c.split("::")[0] for c in cids]
    by_tick: dict[str, list[int]] = {}
    for i, t in enumerate(tick_of):
        by_tick.setdefault(t, []).append(i)

    flat, owner = [], []
    for _, r in df.iterrows():
        refs = r["references"]
        if refs is None or len(refs) == 0:
            continue
        for x in refs:
            s = str(x).strip()
            if s:
                flat.append(s)
                owner.append(str(r["_id"]))
    rmat = vec.transform(flat)
    print(f"references: {len(flat)}")

    # ------------------------------------- PASS A: resolve each question to one filing
    print("\nPASS A: resolving question -> filing...")
    qfile_score: dict[str, dict[str, float]] = {}
    B = 128
    for s in range(0, rmat.shape[0], B):
        sims = linear_kernel(rmat[s:s + B], cmat)
        for i in range(sims.shape[0]):
            row = sims[i]
            n = min(RESOLVE_TOPN, row.shape[0])
            top = np.argpartition(-row, n - 1)[:n]
            acc = qfile_score.setdefault(owner[s + i], {})
            for j in top:
                v = float(row[j])
                if v > 0:
                    acc[tick_of[j]] = acc.get(tick_of[j], 0.0) + v
        print(f"  {min(s + B, rmat.shape[0])}/{rmat.shape[0]}", end="\r")
    print()

    q_tick, ambiguous = {}, []
    for qid, acc in qfile_score.items():
        if not acc:
            continue
        ranked = sorted(acc.items(), key=lambda kv: -kv[1])
        best_t, best_v = ranked[0]
        second_v = ranked[1][1] if len(ranked) > 1 else 0.0
        q_tick[qid] = best_t
        if second_v > 0 and best_v / second_v < AMBIG_MARGIN:
            ambiguous.append(qid)
    print(f"  resolved {len(q_tick)} questions, {len(ambiguous)} ambiguous")

    refs_by_tick: dict[str, list[tuple[str, int]]] = {}
    for k, qid in enumerate(owner):
        t = q_tick.get(qid)
        if t is None:
            continue
        refs_by_tick.setdefault(t, []).append((qid, k))

    # ---------------------------- PASS B: greedy union coverage inside that filing only
    print("\nPASS B: labelling by union coverage...")
    qrels: dict[str, dict[str, int]] = {}
    cover_hit = cover_all = 0

    for n_done, (tick, items) in enumerate(refs_by_tick.items(), 1):
        idxs = by_tick.get(tick, [])
        if not idxs:
            continue
        sub = cmat[idxs]
        ccnt = {j: wcount(ctexts[j]) for j in idxs}

        for qid, k in items:
            ev = wcount(flat[k])
            total = sum(ev.values())
            if total == 0:
                continue
            cover_all += 1

            sims = linear_kernel(rmat[k], sub)[0]
            take = min(FILING_TOPN, len(idxs))
            cand = [idxs[j] for j in np.argpartition(-sims, take - 1)[:take]]

            remaining, covered, picked = ev.copy(), 0, []
            while covered / total < COVER_TARGET:
                best_j, best_gain = None, 0
                for j in cand:
                    gain = sum((remaining & ccnt[j]).values())
                    if gain > best_gain:
                        best_j, best_gain = j, gain
                if best_j is None or best_gain == 0:
                    break
                picked.append((best_j, best_gain))
                remaining = remaining - ccnt[best_j]
                covered += best_gain
                cand.remove(best_j)

            if not picked:
                continue
            if covered / total >= COVER_TARGET:
                cover_hit += 1

            d = qrels.setdefault(qid, {})
            for j, gain in picked:
                g = 2 if gain / total >= GRADE2_SHARE else 1
                d[cids[j]] = max(d.get(cids[j], 0), g)
            top_j = max(picked, key=lambda p: p[1])[0]
            d[cids[top_j]] = 2

        print(f"  {n_done}/{len(refs_by_tick)} filings", end="\r")
    print()

    # -------------------------------------------------------------- invariant check
    for qid, d in qrels.items():
        ticks = {c.split("::")[0] for c in d}
        assert len(ticks) == 1, f"{qid} spans {len(ticks)} filings: {sorted(ticks)[:5]}"
        assert ticks == {q_tick[qid]}, f"{qid} labelled outside its resolved filing"

    # ------------------------------------------------------------------- write out
    with open(OUT / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, d in qrels.items():
            for cid, g in d.items():
                f.write(f"{qid}\t{cid}\t{g}\n")

    with open(OUT / "question_filing.json", "w", encoding="utf-8") as f:
        json.dump({"q_tick": q_tick, "ambiguous": ambiguous}, f, indent=2)

    per_q = [len(v) for v in qrels.values()]
    stats = {
        "tokenizer": TOKENIZER,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "n_chunks": len(cids),
        "n_filings": len(files),
        "n_questions": int(len(df)),
        "n_labelled": len(qrels),
        "labelling": "filing-restricted greedy union coverage, multiplicity-aware",
        "cover_target": COVER_TARGET,
        "grade2_share": GRADE2_SHARE,
        "filing_topn": FILING_TOPN,
        "resolve_topn": RESOLVE_TOPN,
        "refs_reaching_target_pct": float(100 * cover_hit / max(cover_all, 1)),
        "ambiguous_filing_questions": len(ambiguous),
        "evidence_tokens_mean": float(ev_lens.mean()),
        "evidence_tokens_median": float(np.median(ev_lens)),
        "rel_chunks_per_q_mean": float(np.mean(per_q)),
        "rel_chunks_per_q_median": float(np.median(per_q)),
        "max_tokens_observed": int(tl.max()),
    }
    json.dump(stats, open(OUT / "stage1_stats.json", "w"), indent=2)

    print(f"\nDONE -> {OUT}")
    print(f"  chunks              : {len(cids)}")
    print(f"  labelled questions  : {len(qrels)} ({100 * len(qrels) / len(df):.1f}%)")
    print(f"  rel chunks/query    : mean {np.mean(per_q):.2f}  median {np.median(per_q):.0f}")
    print(f"  refs hitting {COVER_TARGET:.0%} cover: {100 * cover_hit / max(cover_all, 1):.1f}%")
    print(f"  ambiguous filing    : {len(ambiguous)}")


if __name__ == "__main__":
    main()