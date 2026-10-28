"""
Why did we score 1 on 48 queries? Join judge scores to retrieval success.

THE QUESTION THIS ANSWERS

Our judge scores are bimodal: 87 queries at or below 2, 108 at or above 8, very
little between. Two explanations fit that shape and they demand opposite fixes:

  A. RETRIEVAL CEILING. The gold evidence was never in the context, so the
     generator had nothing to work from. Fix: deeper context, better routing.
     Nothing about the generator needs to change.

  B. GENERATION FAILURE. The gold evidence WAS in the context and the model
     still got it wrong. Fix: the generator, the prompt, or the training data.

This script decides between them by joining each per-query judge score to
whether any gold chunk appeared in the context the generator actually saw, and
at which rank. If the low scores concentrate in the no-gold bucket, it is A.
If low scores are spread evenly across both buckets, it is B.

It also reports, for the queries where gold was missed at top-3, how deep in
the ranking the first gold chunk sits. That number sets the expected payoff of
simply raising TOPK_CTX: if the gold is usually at rank 4-10, more context is a
cheap and large win. If it is beyond rank 50, retrieval itself is the problem
and more context buys little.

No API cost, no GPU. Seconds to run.

Run:  python -u src/eval/diagnose_scores.py
Output: results/score_diagnosis.md / .json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROC = Path("data/finder/processed_v4")
RESULTS = Path("results")
RUNDIR = RESULTS / "runs"
RUN = RUNDIR / "ft_dense_ticker_rerank_at_50.trec"

TOPK_CTX = 3


def load_qrels(p):
    q = defaultdict(dict)
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, g = line.rstrip("\n").split("\t")
            q[a][b] = int(g)
    return q


def load_run(p):
    run = defaultdict(list)
    for line in open(p, encoding="utf-8"):
        qid, _, cid, rank, _, _ = line.split()
        run[qid].append((int(rank), cid))
    return {q: [c for _, c in sorted(v)] for q, v in run.items()}


def band(vals):
    if not vals:
        return "       n=0"
    a = sorted(vals)
    return (f"n={len(a):>4}  mean {statistics.fmean(a):5.2f}  "
            f"median {a[len(a) // 2]:4.1f}  "
            f"<=2 {100 * sum(1 for x in a if x <= 2) / len(a):4.1f}%  "
            f">=8 {100 * sum(1 for x in a if x >= 8) / len(a):4.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mistral-7b-instruct-v0-3")
    ap.add_argument("--condition", default="retrieved")
    args = ap.parse_args()

    label = f"{args.tag}_{args.condition}"
    jp = RESULTS / f"judge_summary_{label}.json"
    bp = RESULTS / f"hdrr_comparison_{label}.json"
    for p in (jp, bp, RUN):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    ours = {q: float(s) for q, s
            in json.load(open(jp, encoding="utf-8"))["scores"].items()}
    Q = json.load(open(bp, encoding="utf-8"))["queries"]
    qrels = load_qrels(PROC / "qrels_test.tsv")
    run = load_run(RUN)

    rows = {}
    for qid, score in ours.items():
        gold = set(qrels.get(qid, {}))
        ranked = run.get(qid, [])
        ctx = ranked[:TOPK_CTX]

        first_gold = next((i + 1 for i, c in enumerate(ranked) if c in gold),
                          None)
        n_in_ctx = sum(1 for c in ctx if c in gold)
        rows[qid] = {
            "score": score,
            "n_gold": len(gold),
            "gold_in_context": n_in_ctx,
            "first_gold_rank": first_gold,
            "abstained": bool(Q.get(qid, {}).get("abstained_footer")
                              or Q.get(qid, {}).get("abstained_prose")),
            "their_score": Q.get(qid, {}).get("their_hybrid_score"),
        }

    hit = [r["score"] for r in rows.values() if r["gold_in_context"] > 0]
    miss = [r["score"] for r in rows.values() if r["gold_in_context"] == 0]
    full = [r["score"] for r in rows.values()
            if r["n_gold"] and r["gold_in_context"] >= r["n_gold"]]

    their_hit = [r["their_score"] for r in rows.values()
                 if r["gold_in_context"] > 0 and r["their_score"] is not None]
    their_miss = [r["their_score"] for r in rows.values()
                  if r["gold_in_context"] == 0 and r["their_score"] is not None]

    L = ["# Why the scores are bimodal", "",
         f"Joining each per-query judge score to whether gold evidence "
         f"actually reached the generator's top-{TOPK_CTX} context.", "",
         "## Score by retrieval outcome", "",
         "| context | " + "n | mean | median | <=2 | >=8 |",
         "|---|---|---|---|---|---|"]

    def row(nm, vals):
        if not vals:
            return f"| {nm} | 0 | - | - | - | - |"
        a = sorted(vals)
        return (f"| {nm} | {len(a)} | {statistics.fmean(a):.2f} | "
                f"{a[len(a) // 2]:.1f} | "
                f"{100 * sum(1 for x in a if x <= 2) / len(a):.1f}% | "
                f"{100 * sum(1 for x in a if x >= 8) / len(a):.1f}% |")

    L.append(row("at least one gold chunk", hit))
    L.append(row("ALL gold chunks", full))
    L.append(row("no gold chunk", miss))

    print(f"queries analysed: {len(rows)}\n")
    print(f"  gold reached context : {band(hit)}")
    print(f"  ALL gold in context  : {band(full)}")
    print(f"  gold MISSED entirely : {band(miss)}")

    if their_hit and their_miss:
        L += ["", "## Their hybrid on the same split", "",
              f"Their system does not use our retrieval, so this is a control: "
              f"if their scores also collapse on the queries where OUR "
              f"retrieval missed, those queries are simply hard. If their "
              f"scores hold up, the gap is ours to fix.", "",
              "| our context | their mean |", "|---|---|",
              f"| gold reached | {statistics.fmean(their_hit):.2f} |",
              f"| gold missed | {statistics.fmean(their_miss):.2f} |"]
        print(f"\n  their hybrid, gold reached: "
              f"{statistics.fmean(their_hit):5.2f}")
        print(f"  their hybrid, gold missed : "
              f"{statistics.fmean(their_miss):5.2f}")

    # ------------------------------------------- how deep is the missed gold
    depths = [r["first_gold_rank"] for r in rows.values()
              if r["gold_in_context"] == 0 and r["first_gold_rank"]]
    never = sum(1 for r in rows.values()
                if r["gold_in_context"] == 0 and not r["first_gold_rank"])

    buckets = [(4, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
    counts = {f"{a}-{b}": sum(1 for d in depths if a <= d <= b)
              for a, b in buckets}

    L += ["", f"## Where the missed gold actually sits", "",
          f"{len(depths) + never} queries had no gold in the top-{TOPK_CTX}. "
          f"Rank of the first gold chunk for those:", "",
          "| rank of first gold | queries | recoverable by |",
          "|---|---|---|"]
    recover = {"4-5": "TOPK_CTX=5", "6-10": "TOPK_CTX=10",
               "11-20": "TOPK_CTX=20", "21-50": "deeper reranking",
               "51-100": "retrieval, not context"}
    for k, v in counts.items():
        L.append(f"| {k} | {v} | {recover[k]} |")
    L.append(f"| not in top-100 | {never} | retrieval failure |")

    print(f"\n  missed-gold queries: {len(depths) + never}")
    for k, v in counts.items():
        print(f"    first gold at rank {k:>6}: {v:>3}")
    print(f"    not in top-100      : {never:>3}")

    gain10 = sum(v for k, v in counts.items()
                 if k in ("4-5", "6-10"))
    gain20 = gain10 + counts["11-20"]

    L += ["", "## Expected payoff of more context", "",
          f"Raising the generator's context from top-{TOPK_CTX} to top-10 "
          f"would put gold in front of it for **{gain10} additional queries**; "
          f"top-20 would reach **{gain20}**. Those queries currently average "
          f"{statistics.fmean(miss) if miss else 0:.2f}. If they behaved like "
          f"the queries that do get gold "
          f"({statistics.fmean(hit) if hit else 0:.2f}), the overall mean "
          f"would move from "
          f"{statistics.fmean(list(ours.values())):.2f} to roughly "
          f"{(statistics.fmean(list(ours.values())) * len(ours) + gain10 * ((statistics.fmean(hit) if hit else 0) - (statistics.fmean(miss) if miss else 0))) / len(ours):.2f} "
          f"at top-10.", "",
          "That is an upper bound, not a promise: the model was fine-tuned on "
          "roughly two gold chunks per example, so feeding it ten is a "
          "train/test mismatch that may cost some accuracy on queries it "
          "already answers well. It has to be measured, not assumed.", ""]

    out = "\n".join(L) + "\n"
    (RESULTS / "score_diagnosis.md").write_text(out, encoding="utf-8")
    json.dump({"per_query": rows, "depth_counts": counts,
               "never_retrieved": never,
               "mean_gold_hit": statistics.fmean(hit) if hit else None,
               "mean_gold_miss": statistics.fmean(miss) if miss else None},
              open(RESULTS / "score_diagnosis.json", "w"), indent=2)

    print(f"\n  raising TOPK_CTX to 10 would newly cover {gain10} queries")
    print(f"  raising TOPK_CTX to 20 would newly cover {gain20} queries")
    print(f"\nsaved -> results/score_diagnosis.md / .json")


if __name__ == "__main__":
    main()