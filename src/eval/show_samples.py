"""
Print a stratified sample of the comparison set for human reading.

WHY STRATIFIED, NOT RANDOM

A random sample of 15 is mostly cases their system already answers well, which
tells you little. What matters is:

  - queries their hybrid scored 8-10: shows what the rubric rewards, so you can
    judge whether our answer would earn the same
  - queries their hybrid scored 1-3: where they fail and we might not, i.e.
    where our margin would come from
  - our abstentions: the 23 the rubric will score 1-3 regardless. Read these
    hardest - if the context genuinely lacked the answer, abstaining was right
    and the penalty is a rubric artifact worth arguing in the paper. If the
    answer was sitting in the context, it is a real generation failure.

Nothing here costs money. Read it before judging.

Usage:
    python -u src/eval/show_samples.py
    python -u src/eval/show_samples.py --n 5 --width 700
    python -u src/eval/show_samples.py --only abstained
    python -u src/eval/show_samples.py --context        # also dump the chunks
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

RESULTS = Path("results")


def unsan(s: str) -> str:
    """Undo the literal-backslash-n the sanitiser inserted, for reading only."""
    return (s or "").replace("\\n", "\n")


def block(label: str, text: str, width: int, wrap: int = 96) -> str:
    t = unsan(text).strip() or "(empty)"
    if len(t) > width:
        t = t[:width].rstrip() + f" ... [+{len(t) - width} chars]"
    body = "\n".join(textwrap.fill(ln, wrap) if ln.strip() else ""
                     for ln in t.split("\n"))
    pad = "\n".join("    " + ln for ln in body.split("\n"))
    return f"  {label}\n{pad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mistral-7b-instruct-v0-3")
    ap.add_argument("--condition", default="retrieved")
    ap.add_argument("--n", type=int, default=5,
                    help="examples per stratum (default 5)")
    ap.add_argument("--width", type=int, default=450,
                    help="max chars shown per field")
    ap.add_argument("--only", choices=["high", "low", "abstained", "mid"],
                    help="show only one stratum")
    ap.add_argument("--context", action="store_true",
                    help="also print the retrieved chunk ids")
    args = ap.parse_args()

    p = RESULTS / f"hdrr_comparison_{args.tag}_{args.condition}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    data = json.load(open(p, encoding="utf-8"))
    Q = data["queries"]

    def their(qid):
        s = Q[qid].get("their_hybrid_score")
        return -1.0 if s is None else float(s)

    def is_abst(qid):
        return bool(Q[qid].get("abstained_footer")
                    or Q[qid].get("abstained_prose"))

    abstained = sorted([q for q in Q if is_abst(q)], key=their, reverse=True)
    rest = [q for q in Q if not is_abst(q)]
    high = sorted([q for q in rest if their(q) >= 8], key=their, reverse=True)
    low = sorted([q for q in rest if 0 <= their(q) <= 3], key=their)
    mid = sorted([q for q in rest if 4 <= their(q) <= 7], key=their)

    strata = [
        ("THEIR HYBRID SCORED 8-10  (the bar we must clear)", high),
        ("THEIR HYBRID SCORED 1-3   (where our margin could come from)", low),
        ("THEIR HYBRID SCORED 4-7   (contested middle)", mid),
        ("WE ABSTAINED              (rubric will score these 1-3)", abstained),
    ]
    if args.only:
        pick = {"high": 0, "low": 1, "mid": 2, "abstained": 3}[args.only]
        strata = [strata[pick]]

    print(f"model     : {data.get('model')}")
    print(f"condition : {data.get('condition')}")
    print(f"queries   : {data.get('n')}")
    print(f"strata    : high {len(high)}  low {len(low)}  mid {len(mid)}  "
          f"abstained {len(abstained)}")

    for title, pool in strata:
        print("\n" + "=" * 100)
        print(f"{title}   [{len(pool)} total, showing {min(args.n, len(pool))}]")
        print("=" * 100)
        if not pool:
            print("  (none)")
            continue
        # spread across the stratum rather than taking the first n
        step = max(1, len(pool) // max(args.n, 1))
        chosen = pool[::step][:args.n]
        for qid in chosen:
            v = Q[qid]
            sc = v.get("their_hybrid_score")
            print("\n" + "-" * 100)
            print(f"  {qid}    their hybrid score: "
                  f"{'n/a' if sc is None else sc}"
                  f"    footer: {v.get('footer_kind')}"
                  f"{'    TRUNCATED' if v.get('truncated') else ''}")
            print("-" * 100)
            print(block("QUESTION", v["question"], args.width))
            print()
            print(block("GROUND TRUTH", v["ground_truth"], args.width))
            print()
            print(block("OUR ANSWER", v["our_answer"], args.width))
            print()
            print(block("THEIR ANSWER", v["their_hybrid_answer"], args.width))
            if args.context:
                print()
                print(f"    context: {', '.join(v.get('context_cids', []))}")

    print("\n" + "=" * 100)
    print("READING GUIDE")
    print("=" * 100)
    print("""
  For each example ask one question: would their rubric score OUR answer at
  least as high as THEIRS?

  Watch for these specifically:

    - Our answer is correct but far terser than the ground truth. The rubric
      says style is not penalised and extra detail is not rewarded, so terse
      should be fine - confirm it looks that way.
    - Our answer carries the right number but the wrong scale or unit. That is
      a real error under the rubric ("missing critical numerical values").
    - We abstained but the ground truth is plainly present in the context. That
      is a generation failure, not a rubric artifact, and it is fixable.
    - We abstained and the context genuinely does not contain the answer. That
      is retrieval failure plus honest behaviour; the rubric punishes it
      anyway, which is the argument to make in the paper.
    - Their answer is long and hedged while ours is direct. Judges often reward
      the confident specific answer; this is where our margin may live.
""")


if __name__ == "__main__":
    main()