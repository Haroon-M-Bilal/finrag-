"""
Did deeper context help the queries it was supposed to help?

THE QUESTION

Raising the generator's context from top-3 to top-10 moved the mean from 5.54
to 5.82. That +0.28 is an aggregate, and an aggregate can hide two opposite
effects cancelling: queries that gained gold evidence improving, and queries
that already had their evidence getting WORSE because seven additional
distractor chunks were dropped in front of a model fine-tuned on about two.

This splits the per-query change into three groups, using the rank of the first
gold chunk computed in diagnose_scores.py:

  rank 1-3    already had gold at depth 3. Depth 10 adds only distractors here,
              so any drop in this group is pure train/test mismatch - the cost
              of the change.
  rank 4-10   NEWLY covered by depth 10. This is the group the change was made
              for. If it does not improve, more context is not the answer.
  rank >10    still missing gold at depth 10, or never retrieved. Should be
              roughly unchanged; a large move here is noise or a bug.

WHAT THE ANSWER IMPLIES

  Group 4-10 gains, group 1-3 flat      -> depth 10 is a free win; go deeper.
  Group 4-10 gains, group 1-3 loses     -> mismatch confirmed. Retraining the
                                           adapter with 10-chunk contexts
                                           should recover the loss AND keep the
                                           gain. Worth one overnight run.
  Group 4-10 flat                       -> the model cannot exploit evidence
                                           deeper in the context regardless of
                                           training. Context depth is a dead
                                           end; spend the effort on routing.

No API cost, no GPU.

Run:  python -u src/eval/delta_by_depth.py
      python -u src/eval/delta_by_depth.py --a-topk 3 --b-topk 10
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path("results")


def load_scores(tag: str, condition: str, topk: int) -> dict:
    suffix = "" if topk == 3 else f"_k{topk}"
    p = RESULTS / f"judge_summary_{tag}_{condition}{suffix}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return {q: float(s) for q, s
            in json.load(open(p, encoding="utf-8"))["scores"].items()}


def summarise(name, deltas, a_vals, b_vals):
    if not deltas:
        return f"  {name:<34} n=   0"
    improved = sum(1 for d in deltas if d > 0)
    worsened = sum(1 for d in deltas if d < 0)
    return (f"  {name:<34} n={len(deltas):>4}   "
            f"{statistics.fmean(a_vals):5.2f} -> {statistics.fmean(b_vals):5.2f}   "
            f"{statistics.fmean(deltas):+5.2f}   "
            f"up {improved:>3}  down {worsened:>3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mistral-7b-instruct-v0-3")
    ap.add_argument("--condition", default="retrieved")
    ap.add_argument("--a-topk", type=int, default=3)
    ap.add_argument("--b-topk", type=int, default=10)
    args = ap.parse_args()

    a = load_scores(args.tag, args.condition, args.a_topk)
    b = load_scores(args.tag, args.condition, args.b_topk)

    diag = RESULTS / "score_diagnosis.json"
    if not diag.exists():
        raise SystemExit(f"missing {diag} - run diagnose_scores.py first")
    per_q = json.load(open(diag, encoding="utf-8"))["per_query"]

    shared = sorted(set(a) & set(b) & set(per_q))
    print(f"comparing depth {args.a_topk} -> {args.b_topk} "
          f"on {len(shared)} queries\n")

    groups = {
        f"gold already at rank 1-{args.a_topk}": [],
        f"gold NEWLY covered ({args.a_topk + 1}-{args.b_topk})": [],
        f"gold still beyond rank {args.b_topk}": [],
    }
    keys = list(groups)

    for q in shared:
        r = per_q[q].get("first_gold_rank")
        if r is None or r > args.b_topk:
            k = keys[2]
        elif r <= args.a_topk:
            k = keys[0]
        else:
            k = keys[1]
        groups[k].append((q, a[q], b[q]))

    print(f"  {'group':<34} {'n':>6}   {'mean a -> b':^16}  "
          f"{'delta':>6}   movement")
    print("  " + "-" * 92)
    for k in keys:
        rows = groups[k]
        print(summarise(k,
                        [y - x for _, x, y in rows],
                        [x for _, x, _ in rows],
                        [y for _, _, y in rows]))

    overall = [b[q] - a[q] for q in shared]
    print("  " + "-" * 92)
    print(summarise("ALL", overall, [a[q] for q in shared],
                    [b[q] for q in shared]))

    # ------------------------------------------------------------- verdict
    newly = [y - x for _, x, y in groups[keys[1]]]
    already = [y - x for _, x, y in groups[keys[0]]]
    m_new = statistics.fmean(newly) if newly else 0.0
    m_old = statistics.fmean(already) if already else 0.0

    print("\n" + "=" * 94)
    if m_new > 0.3 and m_old > -0.15:
        print("DEEPER CONTEXT IS A CLEAN WIN. The newly covered queries gained "
              f"{m_new:+.2f} and the\nalready-covered ones held at {m_old:+.2f}. "
              "Try depth 20.")
    elif m_new > 0.3 and m_old <= -0.15:
        print(f"MISMATCH CONFIRMED. Newly covered queries gained {m_new:+.2f}, "
              f"but already-covered\nqueries LOST {m_old:+.2f} - the adapter was "
              f"fine-tuned on ~2 chunks and is being\nhanded 10. Retraining with "
              f"10-chunk contexts should keep the gain and recover\nthe loss. "
              f"One overnight QLoRA run.")
    elif m_new <= 0.3:
        print(f"CONTEXT DEPTH IS NOT THE LEVER. Queries that newly received "
              f"their gold evidence\nmoved only {m_new:+.2f}. The model is not "
              f"exploiting evidence deeper in the prompt,\nand retraining is "
              f"unlikely to change that enough to matter. Put the effort into\n"
              f"ROUTING COVERAGE instead - currently 78.4% of queries route, "
              f"and the unrouted\nones are where retrieval collapses.")
    print("=" * 94)

    json.dump({"a_topk": args.a_topk, "b_topk": args.b_topk,
               "n": len(shared),
               "mean_a": statistics.fmean([a[q] for q in shared]),
               "mean_b": statistics.fmean([b[q] for q in shared]),
               "groups": {k: {"n": len(v),
                              "mean_a": statistics.fmean([x for _, x, _ in v]) if v else None,
                              "mean_b": statistics.fmean([y for _, _, y in v]) if v else None,
                              "delta": statistics.fmean([y - x for _, x, y in v]) if v else None}
                          for k, v in groups.items()}},
              open(RESULTS / f"delta_depth_{args.a_topk}_{args.b_topk}.json",
                   "w"), indent=2)
    print(f"\nsaved -> results/delta_depth_{args.a_topk}_{args.b_topk}.json")


if __name__ == "__main__":
    main()