"""
STEP 1 (v4) - numeric target extraction from FinDER gold answers.

HISTORY, because each failure narrowed the problem

v1  Assumed the result appears BEFORE the derivation clause, generalising from
    one example. Arithmetic reproduced only 15.5% of its picks: the dominant
    template is the opposite order (operands first, result last).

v2  Dropped position and searched for whichever figure the arithmetic
    reproduced. Cannot work: arithmetic is INVERTIBLE (reversible) - if
    a - b = c then a = b + c - so with the full derivation present every
    operand is reproducible from the others. 8 of 15 inspected picks were
    wrong (list numerals, the literal 100 in a percentage formula, plain
    operands).

v3  Anchored the pick linguistically ("= 1.97", "yielding $1,712") and used
    arithmetic to confirm. Correct in structure, and most picks became right.
    But two problems remained, both visible in the samples:
      - VERIFICATION WAS TOO WEAK. Any pair drawn from the whole answer, under
        ~10 candidate operations, at 2% tolerance, matches something almost
        always. 691 of 822 anchored picks "verified", which is too many to be
        a filter. "Division -> 3" passed that way.
      - ONE TEMPLATE WAS MISSING. "increased by $111.5 million, calculated as
        539.2 minus 427.7" states the result BEFORE the marker, so v3 left it
        unresolved - the very example v1 was built on. Both orders occur.

v4 FIXES BOTH

  1. TWO ANCHOR RULES, tried in order.
     POST: the figure immediately AFTER a result-marker ("=", "yields",
           "totaling", "results in", "for a total of").
     PRE:  the figure immediately BEFORE a derivation-marker ("calculated as",
           "computed as", "which is", "i.e."), as in "increased by $111.5
           million, calculated as ...".
     Both must sit within a short character window of the marker, so a figure
     a sentence away is never picked up.

  2. LOCAL VERIFICATION. The confirming operands must come from the SAME
     CLAUSE as the target (a bounded character window around it), not from
     anywhere in the answer, and the tolerance is tightened to 0.5% with a
     separate rounded tier at 2%. A coincidental match across a 10-figure
     answer no longer counts as confirmation.

  3. UNIT SANITY. If the target's kind (percent / currency / count) disagrees
     with what the question type implies - a Division or margin question whose
     answer is tagged as a raw count, for instance - the pick is demoted to
     `anchored_only` rather than trusted. This is what "Compositional ->
     1.72259e+06" (a sales figure standing in for a margin) needed.

TIERS

  anchored_verified  anchor found AND local operands reproduce it   (trusted)
  single_figure      one figure in the answer, unambiguous          (trusted)
  anchored_only      anchor found, not locally confirmed            (inspect)
  unresolved         no anchor found                                (excluded)

Only the two trusted tiers should feed a numeric exact-match metric without
inspection, and that subset - not the total - is the honest denominator.

Run:  python -u src/eval/extract_numeric.py
Output: data/finder/processed_v4/answers_numeric.jsonl
        results/numeric_extraction_report.md
"""
from __future__ import annotations

import json
import re
from collections import Counter
from itertools import permutations
from pathlib import Path

import pandas as pd

FINDER = Path("data/finder")
PARQUET = FINDER / "train-00000-of-00001.parquet"
OUT = FINDER / "processed_v4"
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

TOL_TIGHT = 0.005      # 0.5%
TOL_ROUND = 0.02       # 2%, absorbs FinDER's own rounding
ANCHOR_WINDOW = 22     # chars between marker and figure
LOCAL_WINDOW = 190     # chars around the target searched for its operands
MAX_FIGS = 18

SCALE = {"thousand": 3, "thousands": 3, "k": 3,
         "million": 6, "millions": 6, "m": 6, "mm": 6,
         "billion": 9, "billions": 9, "bn": 9, "b": 9,
         "trillion": 12, "trillions": 12, "tn": 12}
UNIT_TAG = {"currency": "USD", "percent": "PCT", "pp": "PP",
            "bps": "BPS", "ratio": "RATIO", "count": "COUNT"}
SCALE_TAG = {0: "unit", 3: "thousand", 6: "million",
             9: "billion", 12: "trillion"}

POST_MARK = re.compile(
    r"(=|\byields?\b|\byielding\b|\bequals?\b|\btotal(?:ing|ling)?\b|"
    r"\bresults? in\b|\bresulting in\b|\bgiving\b|\bgives\b|"
    r"\bfor a total of\b|\bcomes? to\b|\bamounts? to\b|\bworks out to\b|"
    r"\bwe get\b|\bwe obtain\b|\u2248)", re.I)

PRE_MARK = re.compile(
    r"(,?\s*(?:which is\s+)?calculated as\b|,?\s*computed as\b|"
    r",?\s*derived as\b|,?\s*determined as\b|,?\s*i\.e\.|"
    r",?\s*by (?:subtracting|adding|dividing|multiplying)\b)", re.I)

LISTNUM = re.compile(r"(?:^|\n)\s*(?:step\s+)?\d{1,2}\s*[.):]", re.I)
STEPWORD = re.compile(r"\bstep\s+\d{1,2}\b", re.I)
RATIO_CTX = re.compile(r"\d\s*:\s*\d")

NUM = re.compile(r"""
    (?P<paren>\()?
    \s*(?P<cur>[$€£])?\s*
    (?P<sign>-|\u2212)?
    (?P<mant>\d[\d,]*(?:\.\d+)?)
    \s*
    (?P<unit>%|percent|percentage\ points|pp|bps|basis\ points|
             thousands?|millions?|billions?|trillions?|
             k|mm|bn|tn|x)?
    (?P<close>\))?
""", re.I | re.X)

# what kind of answer each question type implies
EXPECT = {"division": {"percent", "ratio", "count", "currency"},
          "multiplication": {"currency", "count", "ratio"},
          "subtract": {"currency", "count", "percent", "pp"},
          "subtraction": {"currency", "count", "percent", "pp"},
          "addition": {"currency", "count"},
          "compositional": {"percent", "currency", "count", "ratio", "pp"}}


def masked(text: str):
    return ([m.span() for m in LISTNUM.finditer(text)]
            + [m.span() for m in STEPWORD.finditer(text)])


def parse_figures(text: str):
    in_k = bool(re.search(r"in thousands", text, re.I))
    skip = masked(text)
    out = []
    for m in NUM.finditer(text):
        p = m.start("mant")
        if any(a <= p < b for a, b in skip):
            continue
        # "3:1" style ratios are not standalone figures
        if RATIO_CTX.search(text[max(0, p - 3):m.end() + 3]):
            continue
        try:
            mant = float(m.group("mant").replace(",", ""))
        except ValueError:
            continue
        unit = (m.group("unit") or "").strip().lower()
        neg = bool(m.group("sign")) or bool(m.group("paren") and m.group("close"))

        if unit in ("%", "percent"):
            kind, exp = "percent", 0
        elif unit in ("pp", "percentage points"):
            kind, exp = "pp", 0
        elif unit in ("bps", "basis points"):
            kind, exp = "bps", 0
        elif unit == "x":
            kind, exp = "ratio", 0
        else:
            exp = SCALE.get(unit, 0)
            kind = "currency" if m.group("cur") else "count"
            if exp == 0 and in_k and kind == "currency":
                exp = 3

        s = -mant if neg else mant
        out.append({"mantissa": s, "scale": exp, "kind": kind,
                    "value": s * (10 ** exp), "text": m.group(0).strip(),
                    "start": p, "end": m.end()})
    return out


def is_year(f) -> bool:
    return (f["kind"] == "count" and f["scale"] == 0
            and float(f["mantissa"]).is_integer()
            and 1900 <= f["mantissa"] <= 2100)


def is_formula_100(f, text: str) -> bool:
    if f["kind"] != "count" or f["value"] != 100:
        return False
    ctx = text[max(0, f["start"] - 12):f["end"] + 12]
    return bool(re.search(r"[x*\u00d7]\s*100|100\s*[x*\u00d7]|/\s*100", ctx))


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def combos(a: float, b: float, qtype: str):
    t = (qtype or "").lower()
    c = []
    if "subtract" in t:
        c += [a - b, b - a]
    elif "addition" in t:
        c += [a + b]
    elif "division" in t:
        if b:
            c += [a / b, (a / b) * 100]
        if a:
            c += [b / a, (b / a) * 100]
    elif "multiplication" in t:
        c += [a * b]
    else:
        c += [a - b, b - a, a + b, a * b]
        if b:
            c += [a / b, (a / b) * 100, ((a - b) / b) * 100]
        if a:
            c += [b / a, (b / a) * 100, ((b - a) / a) * 100]
    return [x for x in c if x is not None and abs(x) < 1e15]


def anchor(text: str, figs):
    """(figure, rule) using POST markers first, then PRE markers."""
    best = None
    for m in POST_MARK.finditer(text):
        after = [f for f in figs if f["start"] >= m.end()]
        if not after:
            continue
        near = min(after, key=lambda f: f["start"])
        if near["start"] - m.end() <= ANCHOR_WINDOW:
            if best is None or near["start"] > best["start"]:
                best = near
    if best is not None:
        return best, "post"

    for m in PRE_MARK.finditer(text):
        before = [f for f in figs if f["end"] <= m.start()]
        if not before:
            continue
        near = max(before, key=lambda f: f["end"])
        if m.start() - near["end"] <= ANCHOR_WINDOW:
            if best is None or near["end"] > best["end"]:
                best = near
    return (best, "pre") if best is not None else (None, None)


def verify_local(target, figs, qtype: str, tol: float):
    """Operands must live in the same clause as the target."""
    lo = target["start"] - LOCAL_WINDOW
    hi = target["end"] + LOCAL_WINDOW
    local = [f for f in figs
             if f is not target and lo <= f["start"] <= hi]
    for a, b in permutations(local, 2):
        for c in combos(a["value"], b["value"], qtype):
            if rel(target["value"], c) <= tol:
                return True
    return False


def unit_ok(target, qtype: str) -> bool:
    return target["kind"] in EXPECT.get((qtype or "").lower(),
                                        set(UNIT_TAG))


def fmt_block(f) -> str:
    if f is None:
        return "N/A"
    return (f"{f['mantissa']:g} | {SCALE_TAG.get(f['scale'], 'unit')} | "
            f"{UNIT_TAG[f['kind']]}")


def main():
    df = pd.read_parquet(PARQUET)
    n_num_q = int((df["type"].notna()
                   & (df["type"].astype(str) != "None")).sum())
    print(f"questions: {len(df)}   numeric subset: {n_num_q} "
          f"({100 * n_num_q / len(df):.1f}%)")

    rows, stat, rules = [], Counter(), Counter()
    per_type = {}

    for _, r in df.iterrows():
        ans = str(r["answer"]) if r["answer"] is not None else ""
        qtype = str(r["type"]) if r["type"] is not None else ""
        is_num = bool(qtype) and qtype != "None"

        rec = {"_id": str(r["_id"]), "type": qtype,
               "category": str(r["category"]),
               "reasoning": bool(r["reasoning"]), "answer": ans,
               "numeric": is_num, "tier": "not_numeric", "rule": None,
               "n_figures": 0, "target": None, "answer_block": "N/A"}

        if is_num:
            d = per_type.setdefault(qtype, Counter())
            d["n"] += 1
            figs = [f for f in parse_figures(ans)
                    if not is_year(f) and not is_formula_100(f, ans)][:MAX_FIGS]
            rec["n_figures"] = len(figs)

            tgt, tier, rule = None, "unresolved", None
            if len(figs) == 1:
                tgt, tier = figs[0], "single_figure"
            elif figs:
                tgt, rule = anchor(ans, figs)
                if tgt is not None:
                    if not unit_ok(tgt, qtype):
                        tier = "anchored_only"
                    elif verify_local(tgt, figs, qtype, TOL_TIGHT):
                        tier = "anchored_verified"
                    elif verify_local(tgt, figs, qtype, TOL_ROUND):
                        tier = "anchored_verified"
                    else:
                        tier = "anchored_only"

            rec.update(tier=tier, rule=rule)
            if tgt is not None:
                rec["target"] = {k: tgt[k] for k in
                                 ("mantissa", "scale", "kind", "value", "text")}
                rec["answer_block"] = fmt_block(tgt)
            d[tier] += 1
            stat[tier] += 1
            if rule:
                rules[rule] += 1
        rows.append(rec)

    with open(OUT / "answers_numeric.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    trusted = stat["anchored_verified"] + stat["single_figure"]
    L = ["# Numeric extraction report (v4)", "",
         f"Numeric subset (FinDER's `type` field): **{n_num_q}** of "
         f"{len(df)} ({100 * n_num_q / len(df):.1f}%)", "",
         "## Confidence tiers", "", "| tier | n | share |", "|---|---|---|",
         f"| anchored_verified | {stat['anchored_verified']} | "
         f"{100 * stat['anchored_verified'] / n_num_q:.1f}% |",
         f"| single_figure | {stat['single_figure']} | "
         f"{100 * stat['single_figure'] / n_num_q:.1f}% |",
         f"| anchored_only | {stat['anchored_only']} | "
         f"{100 * stat['anchored_only'] / n_num_q:.1f}% |",
         f"| unresolved | {stat['unresolved']} | "
         f"{100 * stat['unresolved'] / n_num_q:.1f}% |", "",
         f"**Trusted (verified + single): {trusted} "
         f"({100 * trusted / n_num_q:.1f}%).**", "",
         f"Anchor rule used: POST-marker {rules['post']}, "
         f"PRE-marker {rules['pre']}. Both orders occur in FinDER; v3 handled "
         f"only the first and left the second unresolved.", "",
         "Verification requires the confirming operands to sit in the SAME "
         "CLAUSE as the target. v3 allowed any pair from the whole answer, "
         "which at 2% tolerance over ~10 operations matched almost always and "
         "so confirmed nothing.", "",
         "## By question type", "",
         "| type | n | verified | single | anchor only | unresolved |",
         "|---|---|---|---|---|---|"]
    for t, d in sorted(per_type.items(), key=lambda kv: -kv[1]["n"]):
        L.append(f"| {t} | {d['n']} | {d['anchored_verified']} | "
                 f"{d['single_figure']} | {d['anchored_only']} | "
                 f"{d['unresolved']} |")

    for tier, head, note in [
            ("anchored_verified", "Trusted samples",
             "Each must show the STATED RESULT, not one of its inputs."),
            ("anchored_only", "Anchored but not locally confirmed",
             "Inspect: correct picks the local check missed, plus genuine "
             "misses."),
            ("unresolved", "Unresolved", "No anchor found; excluded.")]:
        L += ["", f"## {head}", "", note, ""]
        shown = 0
        for r in rows:
            if r["tier"] == tier and r["n_figures"] >= 2 and shown < 12:
                tag = f" `{r['rule']}`" if r["rule"] else ""
                L.append(f"- `{r['type']}`{tag} \u2192 "
                         f"**{r['answer_block']}**  \n"
                         f"  _{r['answer'][:180].strip()}_")
                shown += 1

    out = "\n".join(L) + "\n"
    (RESULTS / "numeric_extraction_report.md").write_text(out, encoding="utf-8")
    print("\n" + out)
    print(f"saved -> {OUT / 'answers_numeric.jsonl'}")


if __name__ == "__main__":
    main()