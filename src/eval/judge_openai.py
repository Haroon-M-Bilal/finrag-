"""
GPT-4.1 judge, replicating Cheng et al.'s benchmark grader exactly.

EVERY SETTING BELOW IS COPIED FROM THEIR CODE, NOT CHOSEN BY US:

  model               gpt-4.1              main_benchmark_rag_result.DEFAULT_MODEL
  temperature         0.0                  InferenceParameters(temperature=0.0)
  max_tokens          100                  InferenceParameters(max_tokens=100)
  response_format     {"type":"json_object"}
  system prompt       BENCHMARK_EVALUATION_SYSTEM_PROMPT   (prompts.py, verbatim)
  user prompt         BENCHMARK_EVALUATION_USER_PROMPT     (prompts.py, verbatim)
  score parsing       int(json["score"]), rejected unless 1 <= score <= 10

Do not "improve" any of these. The entire value of this comparison is that the
grader is theirs. A better rubric would make our number incomparable to the
7.59 we are measuring against.

TWO MODES

  --replicate N
      Re-score N of THEIR OWN hybrid answers, which ship with their published
      scores in the benchmark CSVs. If our scores reproduce theirs, the grader
      is demonstrably identical and no reviewer can attribute a difference to
      our evaluation setup. It also measures judge drift: they ran in May 2026,
      we run months later, and gpt-4.1 is neither perfectly deterministic nor
      a frozen snapshot. Drift measured is drift we can report instead of
      being accused of.

  (default)
      Score OUR answers from results/hdrr_comparison_<tag>_<cond>.json.

COST CONTROL

Nothing is sent until you confirm. The script estimates tokens, prints the
projected cost, and waits. Every score is cached to disk, so a re-run or a
crash never pays twice. --dry-run estimates and exits.

Setup:
    pip install openai
    $env:OPENAI_API_KEY = "sk-..."          # PowerShell, this session only

Usage:
    python -u src/eval/judge_openai.py --replicate 40
    python -u src/eval/judge_openai.py
    python -u src/eval/judge_openai.py --condition gold
    python -u src/eval/judge_openai.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RESULTS = Path("results")
ODIR = Path("external/hdrr/output")
USABLE = RESULTS / "hdrr_usable_queries.txt"

MODEL = "gpt-4.1"
TEMPERATURE = 0.0
MAX_TOKENS = 100
RESPONSE_FORMAT = {"type": "json_object"}

# USD per million tokens. Verify against current OpenAI pricing - this only
# drives the pre-flight estimate, never the calls themselves.
PRICE_IN_PER_M = 2.00
PRICE_OUT_PER_M = 8.00

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ----------------------------------------------------------------------------
# verbatim from external/hdrr/src/prompts.py - DO NOT EDIT
# ----------------------------------------------------------------------------
BENCHMARK_EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator for question-answering systems. Your task is to evaluate the correctness of a RAG (Retrieval-Augmented Generation) system's answer compared to the ground truth answer.

Scoring Guidelines:
- Score 1: Completely wrong answer. The response contains factually incorrect information or is entirely irrelevant to the question.
- Score 2-3: Mostly wrong. The answer has some relevance but contains significant errors or misses most key points.
- Score 4-5: Partially correct. The answer captures some correct information but is missing approximately 50% of the key details or contains notable inaccuracies.
- Score 6-7: Mostly correct. The answer covers the main points but may have minor inaccuracies or miss some secondary details.
- Score 8-9: Almost correct. The answer is substantially correct with only minor omissions or trivial errors.
- Score 10: Fully correct. The answer accurately captures all essential information from the ground truth.

Important considerations:
- Focus on factual correctness, not writing style or formatting
- Slight paraphrasing or different wording that preserves meaning should not be penalized
- Missing critical numerical values or key facts should result in lower scores
- Additional correct information beyond the ground truth should not be penalized
- If the answer says it cannot find the information but the ground truth exists, score should be low (1-3)

You MUST respond with valid JSON only, containing a single key "score" with an integer value from 1 to 10."""

BENCHMARK_EVALUATION_USER_PROMPT = """Please evaluate the following RAG system answer against the ground truth.

**Question:**
{query}

**Ground Truth Answer:**
{ground_truth}

**RAG System Answer:**
{rag_answer}

Evaluate the RAG system answer and provide your score as JSON: {{"score": <1-10>}}"""
# ----------------------------------------------------------------------------

UNICODE_REPLACEMENTS = {
    '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
    '\u2013': '-', '\u2014': '--', '\u00D7': 'x', '\u00F7': '/',
    '\u2212': '-', '\u00B1': '+/-', '\u2026': '...', '\u2022': '-',
    '\u00B0': ' deg', '\u00A0': ' ',
}


def sanitize_text(value) -> str:
    """Their CSV sanitiser. Their judge read sanitised text, so ours must too."""
    if value is None:
        return ''
    s = str(value)
    s = s.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
    for uni, ascii_eq in UNICODE_REPLACEMENTS.items():
        s = s.replace(uni, ascii_eq)
    return s


def build_messages(query: str, ground_truth: str, rag_answer: str):
    return [
        {"role": "system", "content": BENCHMARK_EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": BENCHMARK_EVALUATION_USER_PROMPT.format(
            query=query, ground_truth=ground_truth, rag_answer=rag_answer)},
    ]


def est_tokens(messages) -> int:
    """Rough character/4 estimate. Only used for the cost preview."""
    return sum(len(m["content"]) for m in messages) // 4 + 8


def score_one(client, messages):
    """
    Returns (score, error). Parsing mirrors theirs: JSON, key "score", int,
    range-checked. A reply that does not parse is an error, not a zero - a
    silent zero would corrupt the mean.
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format=RESPONSE_FORMAT,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        return None, f"api error: {e}"

    try:
        score = json.loads(raw).get("score")
        if score is None:
            return None, "no 'score' field"
        score = int(score)
        if not 1 <= score <= 10:
            return None, f"score {score} out of range"
        return score, None
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return None, f"parse error: {e} | raw={raw[:120]}"


def load_items_ours(tag: str, condition: str, suffix: str = ""):
    p = RESULTS / f"hdrr_comparison_{tag}_{condition}{suffix}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}\nRun build_hdrr_comparison.py first.")
    data = json.load(open(p, encoding="utf-8"))
    items = []
    for qid, v in data["queries"].items():
        items.append({
            "qid": qid,
            "query": sanitize_text(v["question"]),
            "ground_truth": v["ground_truth"],
            "rag_answer": v["our_answer"],
            "their_score": v.get("their_hybrid_score"),
            "abstained": bool(v.get("abstained_footer")
                              or v.get("abstained_prose")),
        })
    return items, data.get("model", "unknown")


def load_items_theirs(n: int):
    """Their own hybrid answers, with their published scores, on our subset."""
    if not USABLE.exists():
        raise SystemExit(f"missing {USABLE} - run check_hdrr_overlap.py first")
    usable = {l.strip() for l in open(USABLE) if l.strip()}

    seen, items = set(), []
    for p in sorted(ODIR.glob("benchmark-*hybrid*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                qid = (r.get("query_id") or "").strip()
                sc = (r.get("score") or "").strip()
                ans = (r.get("rag_answer") or "").strip()
                gt = (r.get("ground_truth") or "").strip()
                if (qid not in usable or qid in seen or not sc
                        or not ans or not gt):
                    continue
                seen.add(qid)
                items.append({
                    "qid": qid,
                    "query": (r.get("query") or "").strip(),
                    "ground_truth": gt,
                    "rag_answer": ans,
                    "their_score": float(sc),
                    "abstained": False,
                })
    if not items:
        raise SystemExit(f"no hybrid rows found in {ODIR}")
    items.sort(key=lambda d: d["qid"])          # deterministic selection
    return items[:n], "hdrr-hybrid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, metavar="N",
                    help="re-score N of THEIR answers and compare to their "
                         "published scores")
    ap.add_argument("--model-tag", default="mistral-7b-instruct-v0-3",
                    help="tag used in the comparison filename")
    ap.add_argument("--condition", default="retrieved",
                    choices=["retrieved", "gold"])
    ap.add_argument("--v2", action="store_true",
                    help="the answers were built on the SEC-name routed run")
    ap.add_argument("--topk", type=int, default=3,
                    help="context depth used when the answers were built; "
                         "must match build_hdrr_comparison.py or the wrong "
                         "answer file is scored")
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent requests (theirs defaulted to 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate cost and exit")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    args = ap.parse_args()

    if args.replicate:
        items, who = load_items_theirs(args.replicate)
        label = f"replicate_{len(items)}"
        print(f"MODE: replication check on THEIR hybrid answers "
              f"({len(items)} queries)")
    else:
        suffix = ("" if args.topk == 3 else f"_k{args.topk}") + \
                 ("_v2" if args.v2 else "")
        items, who = load_items_ours(args.model_tag, args.condition, suffix)
        label = f"{args.model_tag}_{args.condition}{suffix}"
        print(f"MODE: scoring OUR answers ({who}, {args.condition}) "
              f"- {len(items)} queries")

    cache_path = RESULTS / f"judge_{label}.json"
    cache = json.load(open(cache_path, encoding="utf-8")) \
        if cache_path.exists() else {}
    # drop any cached failures from an older version so they are retried
    cache = {k: v for k, v in cache.items() if v.get("score") is not None}
    todo = [it for it in items if it["qid"] not in cache]
    failures: list[tuple[str, str]] = []

    msgs = {it["qid"]: build_messages(it["query"], it["ground_truth"],
                                      it["rag_answer"]) for it in todo}
    tok_in = sum(est_tokens(m) for m in msgs.values())
    tok_out = len(todo) * 12
    cost = tok_in / 1e6 * PRICE_IN_PER_M + tok_out / 1e6 * PRICE_OUT_PER_M

    print(f"\ncached: {len(items) - len(todo)}   to score: {len(todo)}")
    print(f"estimated input tokens : {tok_in:,}")
    print(f"estimated cost         : ${cost:.2f}  "
          f"(at ${PRICE_IN_PER_M}/M in, ${PRICE_OUT_PER_M}/M out)")

    if args.dry_run:
        print("\ndry run - nothing sent.")
        return
    if todo:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "\nOPENAI_API_KEY is not set.\n"
                '  $env:OPENAI_API_KEY = "sk-..."')
        if not args.yes:
            if input(f"\nspend ~${cost:.2f}? [y/N] ").strip().lower() != "y":
                print("aborted - nothing sent.")
                return

        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("pip install openai")
        client = OpenAI()

        # fail fast on a bad key: one probe before firing N requests at a 401
        probe_s, probe_e = score_one(client, msgs[todo[0]["qid"]])
        if probe_s is None and "401" in str(probe_e):
            raise SystemExit(
                f"\nAPI key rejected (nothing was charged).\n  {probe_e}\n\n"
                f"Check: $env:OPENAI_API_KEY.Length  - a sk-proj key is ~160 "
                f"chars.\nRe-copy the whole key and set it again in THIS "
                f"PowerShell window.")

        lock = threading.Lock()
        done = [0]
        if probe_s is not None:
            cache[todo[0]["qid"]] = {"score": probe_s, "error": None}
            todo = todo[1:]

        def work(it):
            s, err = score_one(client, msgs[it["qid"]])
            return it["qid"], s, err

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for fut in as_completed(futs):
                qid, s, err = fut.result()
                with lock:
                    # only cache successes - a cached failure would be skipped
                    # on the next run and silently vanish from the results
                    if s is not None:
                        cache[qid] = {"score": s, "error": None}
                    else:
                        failures.append((qid, err))
                    done[0] += 1
                    if done[0] % 10 == 0 or done[0] == len(todo):
                        json.dump(cache, open(cache_path, "w"), indent=2)
                        print(f"  {done[0]}/{len(todo)}", end="\r")
        print()
        json.dump(cache, open(cache_path, "w"), indent=2)

    # ------------------------------------------------------------- summarise
    ours = {it["qid"]: cache[it["qid"]]["score"] for it in items
            if cache.get(it["qid"], {}).get("score") is not None}
    errs = failures

    print(f"\n{'=' * 64}")
    print(f"scored {len(ours)}/{len(items)}   failed {len(errs)}")
    if errs:
        for q, e in errs[:5]:
            print(f"  {q}: {e}")

    if not ours:
        raise SystemExit("no scores - nothing to summarise")

    v = list(ours.values())
    print(f"\n  mean score      : {statistics.fmean(v):.2f} / 10")
    print(f"  failure (<=1)   : {100 * sum(1 for x in v if x <= 1) / len(v):.1f}%")
    print(f"  correct  (>=8)  : {100 * sum(1 for x in v if x >= 8) / len(v):.1f}%")

    dist = {}
    for x in v:
        dist[x] = dist.get(x, 0) + 1
    print("\n  distribution:")
    for s in sorted(dist):
        print(f"    {s:2d}: {dist[s]:3d} {'#' * dist[s]}")

    if args.replicate:
        pairs = [(ours[it["qid"]], it["their_score"]) for it in items
                 if it["qid"] in ours and it["their_score"] is not None]
        if pairs:
            diffs = [a - b for a, b in pairs]
            exact = sum(1 for d in diffs if d == 0) / len(diffs)
            within1 = sum(1 for d in diffs if abs(d) <= 1) / len(diffs)
            print(f"\n{'-' * 64}")
            print(f"AGREEMENT WITH THEIR PUBLISHED SCORES (n={len(pairs)})")
            print(f"  exact match        : {100 * exact:.1f}%")
            print(f"  within 1 point     : {100 * within1:.1f}%")
            print(f"  mean difference    : {statistics.fmean(diffs):+.2f} "
                  f"(ours minus theirs)")
            print(f"  mean |difference|  : "
                  f"{statistics.fmean([abs(d) for d in diffs]):.2f}")
            if len(pairs) > 2:
                try:
                    r = statistics.correlation([a for a, _ in pairs],
                                               [b for _, b in pairs])
                    print(f"  correlation        : {r:.3f}")
                except statistics.StatisticsError:
                    pass
            bias = statistics.fmean(diffs)
            print(f"{'-' * 64}")
            if within1 >= 0.80 and abs(bias) <= 0.3:
                print("Grader reproduces theirs. Comparison is sound; cite "
                      "these agreement\nfigures in the paper's methods.")
            elif within1 >= 0.60:
                print(f"Moderate agreement with {bias:+.2f} bias. Usable, but "
                      f"report the bias and\nread it as judge drift, not as a "
                      f"pipeline difference.")
            else:
                print("POOR AGREEMENT. Something differs beyond drift - check "
                      "the prompts and\nmodel string before trusting any "
                      "comparison built on this.")
    else:
        abst = [it["qid"] for it in items if it["abstained"] and it["qid"] in ours]
        if abst:
            av = [ours[q] for q in abst]
            nv = [ours[q] for q in ours if q not in set(abst)]
            print(f"\n{'-' * 64}")
            print(f"ABSTENTION PENALTY UNDER THEIR RUBRIC")
            print(f"  abstained   n={len(av):>4}  mean {statistics.fmean(av):.2f}")
            if nv:
                print(f"  answered    n={len(nv):>4}  mean "
                      f"{statistics.fmean(nv):.2f}")
                lift = statistics.fmean(nv) - statistics.fmean(v)
                print(f"\n  mean excluding abstentions: "
                      f"{statistics.fmean(nv):.2f}  ({lift:+.2f})")
                print(f"  Report BOTH. The headline number is the one over all "
                      f"queries;\n  the second shows how much of the gap is the "
                      f"rubric punishing honesty.")
            print(f"{'-' * 64}")

        theirs_sub = [it["their_score"] for it in items
                      if it["qid"] in ours and it["their_score"] is not None]
        if theirs_sub:
            print(f"\n  their hybrid, same queries: "
                  f"{statistics.fmean(theirs_sub):.2f}  (n={len(theirs_sub)})")
            print(f"  ours                      : {statistics.fmean(v):.2f}")
            print(f"\n  Run compare_hdrr.py for the paired bootstrap - a "
                  f"difference of means\n  without a paired CI is not a result.")

    json.dump({"model": MODEL, "temperature": TEMPERATURE,
               "max_tokens": MAX_TOKENS, "label": label,
               "n_scored": len(ours), "n_failed": len(errs),
               "mean": statistics.fmean(v),
               "fail_rate": sum(1 for x in v if x <= 1) / len(v),
               "correct_rate": sum(1 for x in v if x >= 8) / len(v),
               "distribution": dist, "scores": ours},
              open(RESULTS / f"judge_summary_{label}.json", "w"), indent=2)
    print(f"\nsaved -> results/judge_{label}.json")
    print(f"saved -> results/judge_summary_{label}.json")


if __name__ == "__main__":
    main()