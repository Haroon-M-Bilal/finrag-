"""
Our free router vs their LLM router, on identical queries.

WHY THIS IS THE CLEANEST COMPARISON IN THE PROJECT

HDRR extracts a ticker with an LLM call per query
(TICKER_EXTRACTION_SYSTEM_PROMPT in their prompts.py). We do it with a regex
plus SEC's public company-name table and an acronym stoplist - no model, no
API, no per-query cost.

Their intermediate CSVs record what their router actually decided, per query:

    routed_documents   data\\10k\\converted\\MHK.pdf
    routing_status     filename_fallback

So for every query we share with them we can put three decisions side by side -
their LLM router, our old regex, our new SEC-name router - and score all three
against the same ground truth: the filing our qrels place the gold chunks in.

This is a like-for-like comparison of the component, not of whole pipelines, so
it is immune to the generator gap that dominates the end-to-end scores.

Read-only, CPU only, no API.

Run:  python -u src/eval/compare_routers.py
Output: results/router_comparison.md / .json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROC = Path("data/finder/processed_v4")
ODIR = Path("external/hdrr/output")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

TICKRE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,5}\b")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_qrels(p):
    q = defaultdict(dict)
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, g = line.rstrip("\n").split("\t")
            q[a][b] = int(g)
    return q


def ticker_from_path(p: str) -> str | None:
    """'data\\10k\\converted\\MHK.pdf' -> 'MHK'. Multi-doc routes -> None."""
    if not p or not p.strip():
        return None
    parts = [x for x in re.split(r"[;,|]", p) if x.strip()]
    ticks = set()
    for part in parts:
        stem = re.split(r"[\\/]", part.strip())[-1]
        stem = re.sub(r"\.pdf$", "", stem, flags=re.I).strip().upper()
        if stem:
            ticks.add(stem)
    return ticks.pop() if len(ticks) == 1 else None


def main():
    corpus_ticks = {json.loads(l)["_id"].split("::")[0]
                    for l in open(PROC / "corpus.jsonl", encoding="utf-8")}

    queries = {str(q["_id"]): q["text"]
               for q in jl(PROC / "queries_test.jsonl")}
    qrels = load_qrels(PROC / "qrels_test.tsv")
    truth = {q: next(iter(d)).split("::")[0]
             for q, d in qrels.items() if q in queries}

    rv = PROC / "routing_v2.json"
    new_routing = {k: v["ticker"] for k, v in
                   json.load(open(rv, encoding="utf-8"))["routing"].items()}

    # ---- their router, read from the intermediate (pre-judge) CSVs
    theirs, status = {}, {}
    files = [p for p in ODIR.glob("*.csv") if not p.name.startswith("benchmark-")]
    for p in files:
        with open(p, encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            if not rd.fieldnames or "routed_documents" not in rd.fieldnames:
                continue
            system = p.name.split("-Linq")[0]
            for row in rd:
                qid = (row.get("query_id") or "").strip()
                if qid not in truth:
                    continue
                tk = ticker_from_path(row.get("routed_documents", ""))
                theirs.setdefault(system, {})[qid] = tk
                status.setdefault(system, Counter())[
                    (row.get("routing_status") or "none").strip()] += 1

    if not theirs:
        raise SystemExit("no intermediate CSVs with a routed_documents column")

    best = max(theirs, key=lambda s: len(theirs[s]))
    their_route = theirs[best]
    shared = sorted(set(their_route) & set(truth))
    print(f"their systems with routing recorded: {', '.join(sorted(theirs))}")
    print(f"using '{best}'  -  {len(shared)} queries shared with our test split\n")

    def score(fn, name):
        routed = correct = 0
        wrong = []
        for q in shared:
            tk = fn(q)
            if tk:
                routed += 1
                if tk == truth[q]:
                    correct += 1
                elif len(wrong) < 8:
                    wrong.append((q, truth[q], tk))
        return {"name": name, "routed": routed, "correct": correct,
                "n": len(shared),
                "coverage": routed / len(shared),
                "precision": correct / max(routed, 1),
                "end_to_end": correct / len(shared),
                "wrong_samples": wrong}

    def old_regex(q):
        named = set(TICKRE.findall(queries[q])) & corpus_ticks
        return named.pop() if len(named) == 1 else None

    rows = [
        score(lambda q: their_route.get(q), f"HDRR LLM router ({best})"),
        score(old_regex, "ours: regex only"),
        score(lambda q: new_routing.get(q), "ours: SEC names + stoplist"),
    ]

    print(f"{'router':<32}{'coverage':>10}{'precision':>11}{'routed+correct':>16}")
    print("-" * 69)
    for r in rows:
        print(f"{r['name']:<32}{100*r['coverage']:>9.1f}%"
              f"{100*r['precision']:>10.1f}%{100*r['end_to_end']:>15.1f}%")

    agree = sum(1 for q in shared
                if their_route.get(q) and new_routing.get(q)
                and their_route[q] == new_routing[q])
    both = sum(1 for q in shared
               if their_route.get(q) and new_routing.get(q))
    we_only = sum(1 for q in shared
                  if new_routing.get(q) and not their_route.get(q))
    they_only = sum(1 for q in shared
                    if their_route.get(q) and not new_routing.get(q))
    print(f"\nboth routed: {both}   agreed on the filing: {agree} "
          f"({100*agree/max(both,1):.1f}%)")
    print(f"only we routed: {we_only}    only they routed: {they_only}")

    if best in status:
        print(f"\ntheir routing_status breakdown:")
        for k, v in status[best].most_common():
            print(f"  {k:<24} {v}")

    L = ["# Router comparison: LLM extraction vs a free dictionary", "",
         f"Scored on the {len(shared)} held-out test queries that appear in "
         f"both their published runs and our test split. Ground truth is the "
         f"filing our qrels place the gold chunks in. This compares the "
         f"routing component alone, so it is unaffected by the generator gap "
         f"that dominates end-to-end scores.", "",
         "| router | cost per query | coverage | precision | routed & correct |",
         "|---|---|---|---|---|"]
    costs = ["1 LLM call", "none", "none"]
    for r, c in zip(rows, costs):
        L.append(f"| {r['name']} | {c} | {100*r['coverage']:.1f}% | "
                 f"{100*r['precision']:.1f}% | {100*r['end_to_end']:.1f}% |")
    L += ["", f"Where both routers committed to a filing ({both} queries) they "
          f"agreed {100*agree/max(both,1):.1f}% of the time. Our router "
          f"committed on {we_only} queries theirs left unrouted; theirs "
          f"committed on {they_only} ours left unrouted.", ""]

    for r in rows:
        if r["wrong_samples"]:
            L += [f"### {r['name']} - sample misroutes", "",
                  "| query | gold filing | chosen |", "|---|---|---|"]
            for q, t, c in r["wrong_samples"]:
                L.append(f"| {queries[q][:80].replace('|','/')} | {t} | {c} |")
            L.append("")

    (RESULTS / "router_comparison.md").write_text("\n".join(L) + "\n",
                                                  encoding="utf-8")
    json.dump({"shared": len(shared), "their_system": best, "rows": rows,
               "agreement": {"both_routed": both, "agreed": agree,
                             "ours_only": we_only, "theirs_only": they_only},
               "their_status": dict(status.get(best, {}))},
              open(RESULTS / "router_comparison.json", "w"), indent=2)
    print("\nsaved -> results/router_comparison.md / .json")


if __name__ == "__main__":
    main()
