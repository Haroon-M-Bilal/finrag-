"""
Paired bootstrap restricted to the 853 test queries that ticker routing
actually applies to.

WHY THIS EXISTS

The full-ablation delta for reranking-on-top-of-routing was +0.0075 with a CI
spanning zero, which reads as "no effect". That number is diluted: 235 of the
1088 test queries name no ticker, so ft_dense_ticker falls back to plain dense
retrieval and ft_dense_ticker_rerank@50 reranks that same unrouted list. Those
queries contribute noise to a comparison that is only meaningful where routing
happened.

On the 853 routed queries the effect is clear (0.6000 -> 0.6599). This script
puts a paired confidence interval on it, using one set of resampled query
indices shared across systems so the comparison is paired rather than two
independent intervals.

Run:  python -u src\\eval\\routed_subset_test.py
Output: results/routed_subset_v4.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

PROC = Path("data/finder/processed_v4")
RESULTS = Path("results")
RUNDIR = RESULTS / "runs"
TICKRE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,5}\b")
N_BOOT = 1000
SEED = 0


def load_qrels(p):
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, s = line.rstrip("\n").split("\t")
            q.setdefault(a, {})[b] = int(s)
    return q


def load_trec(p):
    run = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            qid, _, cid, rank, score, _ = line.split()
            run.setdefault(qid, []).append((int(rank), cid))
    return {q: [c for _, c in sorted(v)] for q, v in run.items()}


def ndcg_at(ranked, gold, k=10):
    grades = [gold.get(c, 0) for c in ranked[:k]]
    ideal = sorted(gold.values(), reverse=True)[:k]
    dcg = sum(g / np.log2(r + 1) for r, g in enumerate(grades, 1))
    idcg = sum(g / np.log2(r + 1) for r, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at(ranked, gold, k=10):
    for r, c in enumerate(ranked[:k], 1):
        if gold.get(c, 0) >= 1:
            return 1.0 / r
    return 0.0


def recall_at(ranked, gold, k):
    return sum(1 for c in ranked[:k] if c in gold) / max(len(gold), 1)


def main():
    qrels = load_qrels(PROC / "qrels_test.tsv")
    queries = {}
    for l in open(PROC / "queries_test.jsonl", encoding="utf-8"):
        r = json.loads(l)
        queries[str(r["_id"])] = r["text"]
    ticks = set()
    for l in open(PROC / "corpus.jsonl", encoding="utf-8"):
        ticks.add(json.loads(l)["_id"].split("::")[0])

    routed = [q for q in queries
              if q in qrels and len(set(TICKRE.findall(queries[q])) & ticks) == 1]
    print(f"routed queries: {len(routed)}")

    runs = {p.stem.replace("_at_", "@"): load_trec(p)
            for p in sorted(RUNDIR.glob("*.trec"))}

    systems = ["ft_dense", "small_dense", "base_dense",
               "ft_dense_ticker", "ft_dense_ticker_rerank@50"]
    metrics = {"ndcg@10": lambda r, g: ndcg_at(r, g, 10),
               "mrr@10": lambda r, g: mrr_at(r, g, 10),
               "recall@10": lambda r, g: recall_at(r, g, 10),
               "recall@100": lambda r, g: recall_at(r, g, 100)}

    per_sys = {}
    for s in systems:
        if s not in runs:
            continue
        per_sys[s] = {m: np.array([f(runs[s].get(q, []), qrels[q])
                                   for q in routed])
                      for m, f in metrics.items()}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(routed), size=(N_BOOT, len(routed)))
    boot = {s: {m: v[idx].mean(axis=1) for m, v in d.items()}
            for s, d in per_sys.items()}

    L = [f"# Paired bootstrap on the routed subset ({len(routed)} queries)", "",
         "Restricted to test queries that name exactly one ticker, i.e. the "
         "queries where routing actually applies. The full-set delta for "
         "reranking (+0.0075, CI spanning zero) was diluted by 235 unrouted "
         "queries where both systems fall back to identical plain dense "
         "retrieval.", "",
         "| System | " + " | ".join(metrics) + " | NDCG@10 95% CI |",
         "|" + "---|" * (len(metrics) + 2)]
    for s in systems:
        if s not in per_sys:
            continue
        row = " | ".join(f"{per_sys[s][m].mean():.4f}" for m in metrics)
        b = boot[s]["ndcg@10"]
        L.append(f"| {s} | {row} | [{np.percentile(b, 2.5):.4f}, "
                 f"{np.percentile(b, 97.5):.4f}] |")

    pairs = [("ft_dense_ticker", "ft_dense"),
             ("ft_dense_ticker_rerank@50", "ft_dense_ticker"),
             ("ft_dense_ticker_rerank@50", "ft_dense")]

    L += ["", "## Paired deltas (shared resamples)", "",
          "| A | B | metric | A-B | 95% CI | significant |",
          "|---|---|---|---|---|---|"]
    for a, b in pairs:
        if a not in boot or b not in boot:
            continue
        for m in ("ndcg@10", "mrr@10", "recall@10"):
            d = boot[a][m] - boot[b][m]
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            sig = "yes" if (lo > 0 or hi < 0) else "no"
            L.append(f"| {a} | {b} | {m} | {d.mean():+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {sig} |")

    out = "\n".join(L) + "\n"
    print("\n" + out)
    (RESULTS / "routed_subset_v4.md").write_text(out, encoding="utf-8")
    print("saved -> results/routed_subset_v4.md")


if __name__ == "__main__":
    main()