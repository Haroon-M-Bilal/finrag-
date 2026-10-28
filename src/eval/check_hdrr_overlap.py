"""
Feasibility check for a paired comparison against Cheng et al. (HDRR, 2026).

WHY THIS RUNS FIRST

Their repository publishes fixed 300-query FinDER splits and per-query
LLM-as-judge scores (1-10) for every system in the paper. Because the scores are
per-query and keyed on FinDER's `_id`, a PAIRED comparison on identical
questions is possible - far stronger than placing two aggregate numbers side by
side.

The constraint is leakage. Their queries are drawn from all 497 filings. Our
embedder, reranker and generators were trained on 392 of them, so any query
whose filing is in our TRAIN split cannot be used: we would be scoring models on
filings they were fitted to. Only queries falling in our 98 held-out test
filings are admissible.

FIXES OVER THE FIRST VERSION

  1. SEVEN FILES HAVE NO SYSTEM SEGMENT.
     benchmark-Linq-AI-Research_FinDER-Linq-AI-Research_FinDER-300-01-...
     is their unnamed baseline. The old regex required a system name, so these
     fell into a bucket called "unknown".

  2. TWO JUDGES WERE SILENTLY AVERAGED.  Two of those seven files were scored by
     gpt-5.2, the rest by gpt-4.1. The old version keyed scores by query id
     alone, so the gpt-5.2 files overwrote splits 01 and 02 and the reported
     mean mixed two graders with different calibration. Scores are now keyed on
     (system, judge) and a judge mismatch can no longer collapse.

  3. DUPLICATE QUERY IDS ARE AVERAGED, NOT OVERWRITTEN. Their five 300-query
     splits are not disjoint: 1,354 unique ids across 1,500 slots, so ~146
     queries appear in more than one split. Where a system scored the same query
     twice we average, and report the spread - run-to-run variance of their own
     pipeline plus judge, which is a free reference point for the width of our
     confidence intervals.

  4. JUDGE IS READ FROM THE FILENAME. Two of their systems (index-solution,
     no-rerank-index-solution) have no judge recorded at all; those are marked
     "unspecified" rather than assumed to be gpt-4.1.

Decision rule: above ~250 surviving queries the comparison is worth running;
below ~150 it is too underpowered to carry a claim.

Run:  python -u src/eval/check_hdrr_overlap.py
Output: results/hdrr_overlap.json
        results/hdrr_usable_queries.txt
        results/hdrr_subset_scores.json   <- per-query scores, input to the
                                             paired bootstrap later
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROC = Path("data/finder/processed_v4")
HDRR = Path("external/hdrr")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

QDIR = HDRR / "random_queries_csv"
ODIR = HDRR / "output"

NAME_PREFIX = "benchmark-Linq-AI-Research_FinDER-"
NAME_MID = "Linq-AI-Research_FinDER-300-"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def parse_name(name: str):
    """
    benchmark-Linq-AI-Research_FinDER-[<system>-]Linq-AI-Research_FinDER-300-<NN>-<tail>

    The system segment is OPTIONAL - their unnamed baseline omits it entirely.
    The judge model appears in the tail for most runs but not all.
    Returns (system, judge, split) or None.
    """
    stem = name[:-4] if name.lower().endswith(".csv") else name
    if not stem.startswith(NAME_PREFIX):
        return None
    rest = stem[len(NAME_PREFIX):]
    i = rest.find(NAME_MID)
    if i < 0:
        return None

    system = rest[:i].rstrip("-") or "baseline"
    tail = rest[i + len(NAME_MID):]

    m = re.match(r"(\d+)", tail)
    split = m.group(1) if m else "?"

    jm = re.search(r"gpt-[0-9][0-9.]*", tail)
    judge = jm.group(0) if jm else "unspecified"

    return system, judge, split


def main():
    if not QDIR.exists():
        raise SystemExit(f"not found: {QDIR}\nClone their repo into "
                         f"external/hdrr first.")

    # ------------------------------------------------- their unique query ids
    their_ids, per_group = set(), {}
    for p in sorted(QDIR.glob("*300-0*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            ids = {r["query_id"].strip() for r in csv.DictReader(f)
                   if r.get("query_id")}
        per_group[p.name] = len(ids)
        their_ids |= ids
    slots = sum(per_group.values())
    print(f"their query splits: {len(per_group)} files, {slots} slots, "
          f"{len(their_ids)} unique query ids "
          f"({slots - len(their_ids)} repeats across splits)")

    # --------------------------------------------------- our held-out split
    test = {str(r["_id"]) for r in jl(PROC / "queries_test.jsonl")}
    train = {str(r["_id"]) for r in jl(PROC / "queries_train.jsonl")}
    numeric = {r["_id"]: r for r in jl(PROC / "answers_numeric.jsonl")}
    trusted = {q for q, r in numeric.items()
               if r.get("tier") in ("anchored_verified", "single_figure")}

    usable = sorted(their_ids & test)
    usable_set = set(usable)
    leaked = their_ids & train
    unknown_q = their_ids - test - train

    print(f"\nour split: {len(train)} train / {len(test)} test questions")
    print(f"\n  in our TEST split  (usable)   : {len(usable)}")
    print(f"  in our TRAIN split (leakage)  : {len(leaked)}")
    print(f"  not in our labelled set       : {len(unknown_q)}")
    print(f"  of the usable, numeric-trusted: {len(usable_set & trusted)}")

    # ----------------------------------------- read scores, keyed (sys, judge)
    files = sorted(ODIR.glob("benchmark-*.csv"))
    print(f"\nreading {len(files)} benchmark files...")

    # (system, judge) -> qid -> [scores]
    raw = defaultdict(lambda: defaultdict(list))
    file_map = defaultdict(list)
    unparsed = []

    for p in files:
        parsed = parse_name(p.name)
        if parsed is None:
            unparsed.append(p.name)
            continue
        system, judge, split = parsed
        key = (system, judge)
        file_map[key].append(split)

        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                qid = (r.get("query_id") or "").strip()
                val = (r.get("score") or "").strip()
                if not qid or not val:
                    continue
                try:
                    raw[key][qid].append(float(val))
                except ValueError:
                    continue

    if unparsed:
        print("\nFILENAMES THAT DID NOT PARSE - inspect these:")
        for n in unparsed:
            print(f"  {n}")

    # ------------------------------------------------------------- aggregate
    rows, subset_scores = [], {}
    for (system, judge), d in sorted(raw.items()):
        mean_of = {q: statistics.fmean(v) for q, v in d.items()}
        dups = {q: v for q, v in d.items() if len(v) > 1}
        spread = [max(v) - min(v) for v in dups.values()]

        allv = list(mean_of.values())
        sub = [v for q, v in mean_of.items() if q in usable_set]

        row = {
            "system": system,
            "judge": judge,
            "splits": sorted(file_map[(system, judge)]),
            "n_all": len(allv),
            "n_duplicate_qids": len(dups),
            "mean_abs_spread_on_duplicates":
                float(statistics.fmean(spread)) if spread else None,
            "max_spread_on_duplicates": float(max(spread)) if spread else None,
            "n_subset": len(sub),
        }
        if allv:
            row |= {
                "mean_all": statistics.fmean(allv),
                "fail_all": sum(1 for v in allv if v <= 1) / len(allv),
                "correct_all": sum(1 for v in allv if v >= 8) / len(allv),
            }
        if sub:
            row |= {
                "mean_subset": statistics.fmean(sub),
                "fail_subset": sum(1 for v in sub if v <= 1) / len(sub),
                "correct_subset": sum(1 for v in sub if v >= 8) / len(sub),
            }
            subset_scores[f"{system}|{judge}"] = {
                q: v for q, v in mean_of.items() if q in usable_set}
        rows.append(row)

    # ---------------------------------------------------------------- report
    print("\nScored on ALL their queries  vs  on OUR usable subset:\n")
    print(f"{'system':<26}{'judge':<14}{'n':>5}{'mean':>7}{'fail%':>7}"
          f"{'cor%':>7}  |{'n':>5}{'mean':>7}{'fail%':>7}{'cor%':>7}")
    print("-" * 97)
    for r in rows:
        left = (f"{r['system']:<26}{r['judge']:<14}{r['n_all']:>5}"
                f"{r.get('mean_all', 0):>7.2f}"
                f"{100 * r.get('fail_all', 0):>7.1f}"
                f"{100 * r.get('correct_all', 0):>7.1f}")
        if r["n_subset"]:
            print(left + f"  |{r['n_subset']:>5}{r['mean_subset']:>7.2f}"
                         f"{100 * r['fail_subset']:>7.1f}"
                         f"{100 * r['correct_subset']:>7.1f}")
        else:
            print(left + "  |   no overlap")

    print("\nRun-to-run spread where a system scored the same query twice\n"
          "(their own pipeline + judge variance; a reference for our CI width):\n")
    for r in rows:
        if r["n_duplicate_qids"]:
            print(f"  {r['system']:<26}{r['judge']:<12}"
                  f"{r['n_duplicate_qids']:>4} repeated qids   "
                  f"mean |diff| {r['mean_abs_spread_on_duplicates']:.2f}   "
                  f"max {r['max_spread_on_duplicates']:.1f}")

    # ------------------------------------------------------------- write out
    json.dump({
        "their_unique_query_ids": len(their_ids),
        "their_split_slots": slots,
        "our_test_questions": len(test),
        "our_train_questions": len(train),
        "usable": len(usable),
        "leaked_into_our_train": len(leaked),
        "not_in_our_labelled_set": len(unknown_q),
        "usable_numeric_trusted": len(usable_set & trusted),
        "unparsed_filenames": unparsed,
        "systems": rows,
    }, open(RESULTS / "hdrr_overlap.json", "w"), indent=2)

    with open(RESULTS / "hdrr_usable_queries.txt", "w") as f:
        f.write("\n".join(usable))

    json.dump(subset_scores, open(RESULTS / "hdrr_subset_scores.json", "w"),
              indent=2)

    print(f"\nsaved -> results/hdrr_overlap.json")
    print(f"saved -> results/hdrr_usable_queries.txt ({len(usable)} ids)")
    print(f"saved -> results/hdrr_subset_scores.json "
          f"({len(subset_scores)} system-judge pairs)")

    if len(usable) >= 250:
        print("\nVERDICT: enough queries. Proceed.")
    elif len(usable) >= 150:
        print("\nVERDICT: workable but underpowered - report the CI width "
              "honestly.")
    else:
        print("\nVERDICT: too few queries for a defensible comparison.")


if __name__ == "__main__":
    main()