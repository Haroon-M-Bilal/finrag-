"""
How much of the labelling was corrupted by the acronym/ticker collision?

prepare_finder_v4.py resolved each question to a filing by summed TF-IDF, then
applied this override:

    named = set(re.findall(r"\\b[A-Z][A-Z0-9\\-]{1,5}\\b", qtext)) & valid
    if len(named) == 1:
        q_tick[qid] = t

That is the same rule the router used, and it has the same flaw: CF is cash
flow, NI is net income, IR is investor relations, IP is intellectual property.
Any question using one of those as ordinary financial vocabulary, and naming no
other ticker, was force-labelled to that company - and then Pass B went looking
for gold chunks inside the wrong filing.

This measures the damage three ways:

  1. LABEL CONCENTRATION. How many questions are labelled to each filing. With
     497 filings and N questions, the expected count is N/497. A stoplisted
     ticker sitting far above that is the fingerprint of the collision.

  2. SUSPECT QUESTIONS. Questions whose gold filing is a stoplisted ticker AND
     whose text contains that token AND which name no other ticker - i.e. the
     exact condition under which the override fired wrongly.

  3. COVERAGE QUALITY. For suspects, the share of the reference text the
     selected chunks actually covered. Pass B records nothing per-question, so
     this is recomputed: if a question was sent to the wrong filing, the gold
     chunks it settled for should match the reference text far more weakly than
     a correctly-resolved question does. That separates "collided and landed
     wrong" from "collided but happened to be right" - CF Industries questions
     really do mention CF.

Read-only. CPU only. Does not touch the GPU.

Run:  python -u src/eval/qrels_contamination.py
Output: results/qrels_contamination.md / .json
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PROC = Path("data/finder/processed_v4")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

TICKRE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,5}\b")
TOKRE = re.compile(r"[a-z0-9]+")

# the same stoplist the fixed router uses
STOP = {
    "EPS", "GAAP", "IFRS", "FASB", "SEC", "EBIT", "EBITDA", "COGS", "SGA",
    "DA", "FY", "YOY", "QOQ", "YTD", "MTD", "QTD", "LTM", "NTM", "CAGR",
    "FCF", "NAV", "WACC", "ROI", "ROE", "ROA", "ROIC", "BPS", "DSO", "DPO",
    "IP", "AI", "IT", "HR", "RD", "ESG", "KPI", "TAM", "SAM", "SOM", "SKU",
    "API", "SAAS", "PAAS", "IAAS", "B2B", "B2C", "ARPU", "CAC", "LTV",
    "MRR", "ARR", "NPS", "SSP", "CAPEX", "OPEX", "MA", "IPO", "FX", "PE",
    "PEG", "EV", "SOX", "PCAOB", "ERP", "CRM", "ML", "LLM", "GPU", "CPU",
    "CEO", "CFO", "COO", "CTO", "CIO", "CISO", "CMO", "VP", "EVP", "SVP",
    "BOD", "AGM", "US", "USA", "UK", "EU", "EMEA", "APAC", "LATAM", "NA",
    "ROW", "Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY23", "FY24", "FY22",
    "AND", "FOR", "THE", "NOT", "ALL", "NEW", "KEY", "TOP", "VS", "PER",
    "CF", "NI", "PG", "EXC", "TRV", "MCD", "IR",
}


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


def wc(t):
    return Counter(TOKRE.findall(t.lower()))


def main():
    corpus = {c["_id"]: c["text"] for c in jl(PROC / "corpus.jsonl")}
    all_ticks = {c.split("::")[0] for c in corpus}
    print(f"corpus: {len(corpus)} chunks across {len(all_ticks)} filings")

    report = {}
    lines = ["# qrels contamination from the ticker/acronym collision", "",
             "`prepare_finder_v4.py` resolved questions to filings with the "
             "same regex the router used, so questions using CF, NI, IR, IP "
             "and similar as ordinary financial vocabulary were force-labelled "
             "to those companies and then had gold chunks selected from inside "
             "the wrong filing.", ""]

    for split in ("test", "train"):
        qp, rp = PROC / f"queries_{split}.jsonl", PROC / f"qrels_{split}.tsv"
        if not (qp.exists() and rp.exists()):
            continue
        queries = {str(q["_id"]): q["text"] for q in jl(qp)}
        qrels = load_qrels(rp)
        qrels = {k: v for k, v in qrels.items() if k in queries}

        label = {q: next(iter(d)).split("::")[0] for q, d in qrels.items()}
        counts = Counter(label.values())
        n = len(label)
        expected = n / len(all_ticks)

        # ---- 1. concentration
        top = counts.most_common(12)
        excess = sum(max(0, c - expected) for t, c in counts.items()
                     if t in STOP)

        # ---- 2. suspects
        suspects = []
        for q, t in label.items():
            if t not in STOP:
                continue
            toks = set(TICKRE.findall(queries[q])) & all_ticks
            if t in toks and len(toks) == 1:
                suspects.append(q)

        # ---- 3. coverage quality for suspects vs everyone else
        def coverage(q):
            ev = wc(queries[q])
            tot = sum(ev.values())
            if not tot:
                return None
            got = Counter()
            for cid in qrels[q]:
                if cid in corpus:
                    got += wc(corpus[cid])
            return sum((ev & got).values()) / tot

        sus_cov = [c for c in (coverage(q) for q in suspects) if c is not None]
        others = [q for q in label if q not in set(suspects)]
        oth_cov = [c for c in (coverage(q) for q in others[:2000])
                   if c is not None]

        print(f"\n{'=' * 64}\n{split.upper()}  {n} labelled questions  "
              f"(expected {expected:.1f} per filing)\n{'=' * 64}")
        print("  most-labelled filings:")
        for t, c in top:
            mark = "  <- stoplisted" if t in STOP else ""
            print(f"    {t:<6} {c:>5}   {c / expected:5.1f}x expected{mark}")
        print(f"\n  excess over expectation on stoplisted tickers: "
              f"{excess:.0f} questions ({100 * excess / n:.1f}%)")
        print(f"  questions matching the misfire condition exactly: "
              f"{len(suspects)} ({100 * len(suspects) / n:.1f}%)")
        if sus_cov and oth_cov:
            print(f"\n  query-term coverage by the chosen gold chunks:")
            print(f"    suspects : {statistics.fmean(sus_cov):.3f}  "
                  f"(n={len(sus_cov)})")
            print(f"    others   : {statistics.fmean(oth_cov):.3f}  "
                  f"(n={len(oth_cov)})")
            ratio = statistics.fmean(sus_cov) / max(statistics.fmean(oth_cov),
                                                    1e-9)
            print(f"    ratio    : {ratio:.2f}")
            if ratio < 0.85:
                print("    -> suspects are matched markedly worse. Consistent "
                      "with the\n       labels pointing at the wrong filing.")
            else:
                print("    -> suspects match about as well as everything "
                      "else. Many are\n       probably genuine mentions of "
                      "those companies.")

        report[split] = {
            "n_labelled": n, "expected_per_filing": expected,
            "top_filings": top,
            "excess_on_stoplisted": excess,
            "excess_pct": 100 * excess / n,
            "n_suspects": len(suspects), "suspect_pct": 100 * len(suspects) / n,
            "suspect_coverage": statistics.fmean(sus_cov) if sus_cov else None,
            "other_coverage": statistics.fmean(oth_cov) if oth_cov else None,
            "suspect_ids": suspects[:200],
        }

        lines += [f"## {split}", "",
                  f"{n} labelled questions across {len(all_ticks)} filings "
                  f"(expected {expected:.1f} each).", "",
                  "| filing | questions | x expected | stoplisted |",
                  "|---|---|---|---|"]
        for t, c in top:
            lines.append(f"| {t} | {c} | {c / expected:.1f}x | "
                         f"{'yes' if t in STOP else 'no'} |")
        lines += ["",
                  f"Excess over expectation on stoplisted tickers: "
                  f"**{excess:.0f} questions ({100 * excess / n:.1f}%)**. "
                  f"Questions matching the misfire condition exactly "
                  f"(labelled to a stoplisted ticker, containing that token, "
                  f"naming no other): **{len(suspects)} "
                  f"({100 * len(suspects) / n:.1f}%)**.", ""]
        if sus_cov and oth_cov:
            lines += [f"Query-term coverage by the selected gold chunks is "
                      f"{statistics.fmean(sus_cov):.3f} for suspects against "
                      f"{statistics.fmean(oth_cov):.3f} for the rest.", ""]

    (RESULTS / "qrels_contamination.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    json.dump(report, open(RESULTS / "qrels_contamination.json", "w"),
              indent=2)
    print(f"\nsaved -> results/qrels_contamination.md / .json")


if __name__ == "__main__":
    main()
