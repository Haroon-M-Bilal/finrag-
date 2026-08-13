"""
V2 BLOCKING CHECKS - answers reviewer questions 1 and 2 empirically.

Q1 TRUNCATION: chunk size was chosen in WORD space (512 words). BGE-small and the
   cross-encoder both cap at 512 TOKENS. Financial prose runs >1 token/word, so
   part of every chunk may be silently dropped at encode time. If true, this alone
   could explain "reranking hurts" - the cross-encoder sees query + truncated chunk
   in one 512-token budget, so it may see almost none of the passage.

Q2 ADJACENCY: 5.58 relevant chunks/query is higher than 409-word evidence in
   512-word chunks should produce. Are the extra labels genuine distinct passages,
   or overlap-adjacent near-duplicates (ch12, ch13, ch14) counted separately?
   If mostly adjacent, v2 recall@k rises for reasons unrelated to retrieval quality
   and v1 -> v2 numbers are not comparable.

Run:  python src\\eval\\check_v2.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer

PROC = Path("data/finder/processed_v2")
EMB_MODEL = "BAAI/bge-small-en-v1.5"
RR_MODEL = "BAAI/bge-reranker-base"
N_SAMPLE = 500


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    rng = np.random.default_rng(0)
    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    queries = {str(q["_id"]): q["text"] for q in jl(PROC / "queries.jsonl")}

    # ================= Q1: TOKEN TRUNCATION =================
    print("=" * 62)
    print("Q1. TOKEN LENGTH vs 512-TOKEN ENCODER LIMIT")
    idx = rng.choice(len(ctexts), size=min(N_SAMPLE, len(ctexts)), replace=False)
    samp = [ctexts[i] for i in idx]

    for name, mid in (("embedder (bge-small)", EMB_MODEL),
                      ("reranker (bge-reranker-base)", RR_MODEL)):
        tok = AutoTokenizer.from_pretrained(mid)
        lens = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in samp])
        words = np.array([len(t.split()) for t in samp])
        ratio = float((lens / np.maximum(words, 1)).mean())
        over = int((lens > 512).sum())
        print(f"\n  {name}")
        print(f"    tokens/chunk: mean {lens.mean():.0f}  median {np.median(lens):.0f}"
              f"  90th {np.percentile(lens,90):.0f}  max {lens.max()}")
        print(f"    tokens per word: {ratio:.2f}")
        print(f"    chunks EXCEEDING 512 tokens: {over}/{len(lens)} ({100*over/len(lens):.1f}%)")
        if over:
            lost = np.clip(lens - 512, 0, None)
            frac = float((lost[lens > 512] / lens[lens > 512]).mean())
            print(f"    of those, mean fraction TRUNCATED: {100*frac:.1f}%")

    # cross-encoder: query + chunk share the budget
    tok_rr = AutoTokenizer.from_pretrained(RR_MODEL)
    qs = [queries[k] for k in list(queries)[:200]]
    qlen = np.mean([len(tok_rr(q, add_special_tokens=False)["input_ids"]) for q in qs])
    clen = np.array([len(tok_rr(t, add_special_tokens=False)["input_ids"]) for t in samp])
    budget = 512 - qlen - 4
    print(f"\n  CROSS-ENCODER PAIR BUDGET")
    print(f"    mean query tokens: {qlen:.0f}  -> chunk budget ~{budget:.0f} tokens")
    tr = int((clen > budget).sum())
    print(f"    pairs where chunk is truncated: {tr}/{len(clen)} ({100*tr/len(clen):.1f}%)")
    if tr:
        keep = float((budget / clen[clen > budget]).mean())
        print(f"    of those, reranker sees only ~{100*keep:.0f}% of the chunk")
        print("    => TRUNCATION IS A CONFOUND for any reranking conclusion.")

    # ================= Q2: QREL ADJACENCY =================
    print("\n" + "=" * 62)
    print("Q2. ARE RELEVANT CHUNKS DISTINCT OR OVERLAP-ADJACENT?")
    qrels = defaultdict(dict)
    f = open(PROC / "qrels.tsv", encoding="utf-8"); next(f)
    for line in f:
        q, c, s = line.rstrip("\n").split("\t")
        qrels[q][c] = int(s)

    def parse(cid):
        t, ch = cid.split("::ch")
        return t, int(ch)

    n_rel, n_runs, adj_frac, n_docs = [], [], [], []
    for q, d in qrels.items():
        byfile = defaultdict(list)
        for c in d:
            t, i = parse(c)
            byfile[t].append(i)
        n_rel.append(len(d)); n_docs.append(len(byfile))
        runs = 0; adj = 0
        for t, ixs in byfile.items():
            ixs = sorted(ixs)
            runs += 1
            for a, b in zip(ixs, ixs[1:]):
                if b - a == 1:
                    adj += 1
                else:
                    runs += 1
        n_runs.append(runs)
        adj_frac.append(adj / max(len(d) - 1, 1) if len(d) > 1 else 0.0)

    n_rel = np.array(n_rel); n_runs = np.array(n_runs); n_docs = np.array(n_docs)
    print(f"  queries: {len(n_rel)}")
    print(f"  relevant chunks/query : mean {n_rel.mean():.2f}  median {np.median(n_rel):.0f}")
    print(f"  DISTINCT contiguous runs/query (adjacent chunks merged):")
    print(f"                          mean {n_runs.mean():.2f}  median {np.median(n_runs):.0f}")
    print(f"  distinct filings/query: mean {n_docs.mean():.2f}  median {np.median(n_docs):.0f}")
    print(f"  mean fraction of relevant pairs that are ADJACENT: {np.mean(adj_frac):.3f}")
    print(f"\n  interpretation:")
    if n_runs.mean() <= 2.5:
        print("    Relevant chunks collapse to ~1-2 contiguous spans per query.")
        print("    => The high count IS overlap adjacency, not extra distinct evidence.")
        print("    => Recall@k will be inflated vs v1. Report span-level recall too,")
        print("       and do NOT compare v1 and v2 numbers directly.")
    else:
        print("    Relevant chunks form several separate spans - genuinely multi-passage.")
        print("    => Multi-chunk qrels are justified; counts are not just adjacency.")

    if n_docs.mean() > 1.5:
        print(f"    NOTE: {n_docs.mean():.1f} filings/query on average - evidence spans")
        print("    multiple companies for some questions.")


if __name__ == "__main__":
    main()