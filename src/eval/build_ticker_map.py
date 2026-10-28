"""
Close the routing gap: company names -> tickers, without an LLM.

THE PROBLEM

run_ablation_split.py routes a query to a filing only when exactly one
all-caps token in the query is a ticker in the corpus:

    TICKRE = re.compile(r"\\b[A-Z][A-Z0-9\\-]{1,5}\\b")
    named = set(TICKRE.findall(q["text"])) & all_ticks
    if len(named) == 1: route

That routes 78.4% of queries. The other 21.6% fall back to full-corpus search,
where NDCG@10 is 0.1866 instead of 0.5342. Three distinct failure modes, all
visible in the data:

  1. NAME, NOT TICKER.  "Div policy of Intuit reflects interplay of capital
     returns." Intuit is INTU; the regex sees nothing. HDRR handles this with
     an LLM call per query (TICKER_EXTRACTION_SYSTEM_PROMPT: "If the query
     mentions a company name instead of ticker, convert it to the correct
     ticker symbol"). A dictionary does the same job for free.

  2. ACRONYM COLLISION.  "Growth acceleration in SNPS Design IP segment."
     SNPS is right there - but IP is International Paper's ticker, so the set
     has two members, len(named) != 1, and a query that named its ticker
     explicitly is thrown away. Pure loss, no upside.

  3. SINGLE-LETTER TICKERS.  The regex requires 2-6 characters, so A
     (Agilent), F (Ford) and T (AT&T) can never be routed, even though those
     filings are in the corpus. Widening the regex to one character would be
     catastrophic for precision; a name dictionary fixes it safely.

THE APPROACH

Names come from SEC's own company_tickers.json - public, keyless, and
authoritative. It is restricted to the tickers actually present in the corpus,
then normalised into match variants ("NVIDIA CORP" -> "nvidia", "nvidia corp").

Resolution order, strongest evidence first:
  a. an unambiguous company NAME in the query wins outright. A name is far
     stronger evidence than a bare acronym, so it overrides collisions.
  b. otherwise, ticker tokens that survive the acronym stoplist. Exactly one
     -> route.
  c. anything still ambiguous stays unrouted, exactly as before. This is
     deliberately conservative: a wrong route is worse than no route, because
     it confines retrieval to the wrong company's filing with no escape.

WHY THIS IS MEASURED, NOT ASSERTED

Routing accuracy is checked against qrels. Every labelled query's gold chunks
live in exactly one filing (enforced by an assert in prepare_finder_v4), so the
correct routing target is known. The script reports coverage AND precision for
the old rule and the new one. A rule that routes more queries but routes them
wrong is worse than the rule it replaced, and this is how you tell.

No GPU. No API. Needs one HTTPS request to sec.gov.

Run:  python -u src/eval/build_ticker_map.py
Output: data/finder/processed_v4/ticker_map.json
        data/finder/processed_v4/routing_v2.json
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

PROC = Path("data/finder/processed_v4")
SEC_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_CACHE = PROC / "sec_company_tickers.json"

# SEC asks for a contact in the User-Agent. Theirs is a courtesy requirement,
# not authentication; requests without it get rate-limited or refused.
UA = "FinRAG academic research (haroonmbilal007@gmail.com)"

TICKRE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,5}\b")

# Corporate suffixes stripped when building name variants. Order matters:
# longer forms first so "HOLDINGS INC" does not leave a dangling "HOLDINGS".
# Minimum length for a single-word name variant. "ppl" (PPL Corp) and "key"
# (KeyCorp) collide with ordinary words far more often than they identify the
# company, and any such firm is still reachable through its ticker token.
# Multi-word variants are exempt - "key energy services" is unambiguous.
MIN_NAME_LEN = 4

SUFFIXES = [
    "holdings inc", "holding company", "group inc", "class a", "class b",
    "incorporated", "corporation", "technologies", "international",
    "laboratories", "industries", "enterprises", "companies", "holdings",
    "solutions", "resources", "financial", "worldwide", "company",
    "systems", "holding", "limited", "group", "corp", "inc", "plc",
    "ltd", "co", "sa", "nv", "ag", "the",
]

# Tokens that look like tickers but are financial vocabulary. Each of these is
# ALSO a real ticker, which is exactly why they poison the naive rule.
ACRONYM_STOP = {
    # accounting and reporting
    "EPS", "GAAP", "IFRS", "FASB", "SEC", "EBIT", "EBITDA", "COGS", "SGA",
    "DA", "FY", "YOY", "QOQ", "YTD", "MTD", "QTD", "LTM", "NTM", "CAGR",
    "FCF", "NAV", "WACC", "ROI", "ROE", "ROA", "ROIC", "BPS", "DSO", "DPO",
    # operations and strategy
    "IP", "AI", "IT", "HR", "RD", "ESG", "KPI", "TAM", "SAM", "SOM", "SKU",
    "API", "SAAS", "PAAS", "IAAS", "B2B", "B2C", "ARPU", "CAC", "LTV",
    "MRR", "ARR", "NPS", "SSP", "CAPEX", "OPEX", "MA", "IPO", "FX", "PE",
    "PEG", "EV", "SOX", "PCAOB", "ERP", "CRM", "ML", "LLM", "GPU", "CPU",
    # roles and bodies
    "CEO", "CFO", "COO", "CTO", "CIO", "CISO", "CMO", "VP", "EVP", "SVP",
    "BOD", "AGM",
    # geography and time
    "US", "USA", "UK", "EU", "EMEA", "APAC", "LATAM", "NA", "ROW",
    "Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY23", "FY24", "FY22",
    # misc noise seen in FinDER phrasing
    "AND", "FOR", "THE", "NOT", "ALL", "NEW", "KEY", "TOP", "VS", "PER",
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


def fetch_sec_map() -> dict:
    if SEC_CACHE.exists():
        print(f"using cached {SEC_CACHE}")
        return json.load(open(SEC_CACHE, encoding="utf-8"))
    print(f"downloading {SEC_URL} ...")
    req = urllib.request.Request(SEC_URL, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"\ncould not download the SEC ticker file: {e}\n\n"
            f"Open {SEC_URL} in a browser, save it as\n"
            f"  {SEC_CACHE}\n"
            f"and run this again.\n")
    SEC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(SEC_CACHE, "w"), indent=2)
    print(f"saved -> {SEC_CACHE} ({len(data)} filers)")
    return data


def name_variants(title: str) -> set[str]:
    """'NVIDIA CORP' -> {'nvidia corp', 'nvidia'}  - never shorter than 3 chars."""
    t = title.lower()
    t = re.sub(r"[/\\].*?[/\\]", " ", t)          # strip /DE/, \NEW\ style tags
    t = re.sub(r"[^a-z0-9& ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    def ok(x: str) -> bool:
        return len(x) >= MIN_NAME_LEN or " " in x

    out = set()
    if ok(t):
        out.add(t)
    changed = True
    while changed:
        changed = False
        for suf in SUFFIXES:
            if t.endswith(" " + suf):
                t = t[: -(len(suf) + 1)].strip()
                changed = True
                if ok(t):
                    out.add(t)
    return out


def build_map(corpus_ticks: set[str], sec: dict):
    """ticker -> sorted name variants, restricted to tickers in the corpus."""
    by_ticker = {}
    for row in sec.values():
        tk = str(row.get("ticker", "")).upper().strip()
        if tk in corpus_ticks:
            by_ticker.setdefault(tk, set()).update(
                name_variants(str(row.get("title", ""))))

    # a name that maps to more than one corpus ticker is useless for routing
    owners = defaultdict(set)
    for tk, names in by_ticker.items():
        for n in names:
            owners[n].add(tk)
    dropped = {n for n, t in owners.items() if len(t) > 1}
    if dropped:
        for tk in by_ticker:
            by_ticker[tk] -= dropped

    return {k: sorted(v) for k, v in by_ticker.items() if v}, dropped


def make_router(corpus_ticks: set[str], tmap: dict):
    """Longest-name-first matching, so 'match group' beats 'match'."""
    name_to_tick = {}
    for tk, names in tmap.items():
        for n in names:
            name_to_tick[n] = tk
    ordered = sorted(name_to_tick, key=len, reverse=True)
    patterns = [(re.compile(r"\b" + re.escape(n) + r"\b"), name_to_tick[n])
                for n in ordered]

    def route(text: str):
        """-> (ticker|None, how).  Ticker evidence outranks name evidence."""
        toks = set(TICKRE.findall(text)) - ACRONYM_STOP
        cands = toks & corpus_ticks
        if len(cands) == 1:
            return cands.pop(), "ticker"

        low = text.lower()
        hits, seen = [], set()
        for rx, tk in patterns:
            if tk not in seen and rx.search(low):
                hits.append(tk)
                seen.add(tk)

        if len(cands) > 1:
            # several tickers named; a company name can break the tie
            both = set(hits) & cands
            if len(both) == 1:
                return both.pop(), "ticker+name"
            return None, "ambiguous_ticker"
        if len(hits) == 1:
            return hits[0], "name"
        if len(hits) > 1:
            return None, "ambiguous_name"
        return None, "none"

    return route


def old_route(text: str, corpus_ticks: set[str]):
    named = set(TICKRE.findall(text)) & corpus_ticks
    return (named.pop(), "ticker") if len(named) == 1 else (None, "none")


def evaluate(queries, qrels, router, label):
    """Coverage and precision against the filing the gold chunks actually sit in."""
    routed = correct = wrong = 0
    gold_known = 0
    by_how = defaultdict(lambda: [0, 0])
    misroutes = []

    for q in queries:
        qid = str(q["_id"])
        tk, how = router(q["text"])
        if tk:
            routed += 1
        gold = qrels.get(qid)
        if not gold:
            continue
        gold_known += 1
        truth = next(iter(gold)).split("::")[0]
        if tk:
            by_how[how][0] += 1
            if tk == truth:
                correct += 1
                by_how[how][1] += 1
            else:
                wrong += 1
                if len(misroutes) < 10:
                    misroutes.append((qid, truth, tk, how, q["text"][:110]))

    n = len(queries)
    print(f"\n{label}")
    print(f"  routed            : {routed}/{n}  ({100 * routed / n:.1f}%)")
    if gold_known:
        print(f"  correct filing    : {correct}  "
              f"({100 * correct / max(routed, 1):.1f}% of routed)")
        print(f"  WRONG filing      : {wrong}  "
              f"({100 * wrong / max(routed, 1):.1f}% of routed)")
        print(f"  routed+correct    : {100 * correct / gold_known:.1f}% "
              f"of all labelled queries")
    if by_how:
        print("  by evidence:")
        for how, (tot, ok) in sorted(by_how.items()):
            print(f"    {how:<14} {tot:>5} routed, "
                  f"{100 * ok / max(tot, 1):5.1f}% correct")
    return {"routed": routed, "n": n, "correct": correct, "wrong": wrong,
            "gold_known": gold_known,
            "by_how": {k: {"routed": v[0], "correct": v[1]}
                       for k, v in by_how.items()}}, misroutes


def main():
    corpus = jl(PROC / "corpus.jsonl")
    corpus_ticks = {c["_id"].split("::")[0] for c in corpus}
    del corpus
    print(f"corpus filings: {len(corpus_ticks)}")

    sec = fetch_sec_map()
    tmap, dropped = build_map(corpus_ticks, sec)
    covered = len(tmap)
    print(f"name variants built for {covered}/{len(corpus_ticks)} filings "
          f"({100 * covered / len(corpus_ticks):.1f}%)")
    if dropped:
        print(f"  {len(dropped)} name variants dropped as ambiguous across "
              f"filings")
    missing = sorted(corpus_ticks - set(tmap))
    if missing:
        print(f"  no SEC name for: {', '.join(missing[:15])}"
              f"{' ...' if len(missing) > 15 else ''}")

    router = make_router(corpus_ticks, tmap)

    out = {}
    for split in ("test", "train"):
        qp = PROC / f"queries_{split}.jsonl"
        rp = PROC / f"qrels_{split}.tsv"
        if not qp.exists() or not rp.exists():
            continue
        queries = jl(qp)
        qrels = load_qrels(rp)

        print(f"\n{'=' * 68}\n{split.upper()}  ({len(queries)} queries)\n"
              f"{'=' * 68}")
        old_stats, _ = evaluate(
            queries, qrels, lambda t: old_route(t, corpus_ticks),
            "OLD  regex only")
        new_stats, misroutes = evaluate(
            queries, qrels, router, "NEW  SEC names + acronym stoplist")

        d_routed = new_stats["routed"] - old_stats["routed"]
        d_correct = new_stats["correct"] - old_stats["correct"]
        print(f"\n  CHANGE: {d_routed:+d} routed, {d_correct:+d} "
              f"routed to the correct filing")

        if misroutes:
            print("\n  sample misroutes (gold filing -> what we chose):")
            for qid, truth, tk, how, txt in misroutes[:6]:
                print(f"    {truth:>6} -> {tk:<6} [{how}]  {txt}")

        out[split] = {"old": old_stats, "new": new_stats}

    routing = {}
    for split in ("test", "train"):
        qp = PROC / f"queries_{split}.jsonl"
        if not qp.exists():
            continue
        for q in jl(qp):
            tk, how = router(q["text"])
            if tk:
                routing[str(q["_id"])] = {"ticker": tk, "how": how}

    json.dump(tmap, open(PROC / "ticker_map.json", "w"), indent=2)
    json.dump({"routing": routing, "stats": out,
               "acronym_stoplist": sorted(ACRONYM_STOP),
               "source": SEC_URL},
              open(PROC / "routing_v2.json", "w"), indent=2)

    print(f"\nsaved -> {PROC / 'ticker_map.json'}")
    print(f"saved -> {PROC / 'routing_v2.json'}  ({len(routing)} routed)")

    t = out.get("test", {})
    if t:
        gain = t["new"]["correct"] - t["old"]["correct"]
        if gain >= 30:
            print(f"\n{gain} more TEST queries now route to the right filing. "
                  f"Worth rebuilding\nthe run file - run reroute_and_rerank.py "
                  f"next.")
        elif gain > 0:
            print(f"\nOnly {gain} more TEST queries route correctly. Marginal; "
                  f"check the\nmisroute samples before spending GPU time.")
        else:
            print(f"\nNo improvement. Do not rebuild the run file - inspect "
                  f"the misroutes\nand the stoplist first.")


if __name__ == "__main__":
    main()