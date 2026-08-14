"""
OFFLINE DIAGNOSTICS on the dumped run files. No GPU, no model loading.

Answers three questions the ablation table raises but cannot settle:

  1. IS THE TICKER-ROUTING GAIN CIRCULAR?
     Ticker matching was used at LABELLING time to decide which filing each
     question's gold evidence lives in, and again at RETRIEVAL time to restrict
     the candidate pool. A reviewer will ask whether the +0.3468 gain is an
     artifact of the same signal appearing on both sides.
     Test: split the test set into queries that name a ticker (853) and queries
     that do not (235). The unnamed group had its filing resolved by TF-IDF over
     the reference text, not by ticker, and gets no routing at retrieval time.
     If ft_dense scores similarly on both groups, the LABELS are not
     ticker-dependent and the routing gain is a real retrieval effect.
     Within-filing gold selection came from union coverage against FinDER's own
     reference text in both groups, which is the part the metric actually
     measures.

  2. WHAT DID ROUTING ACTUALLY FIX?
     Decomposes ft_dense failures into "gold was in the wrong-company noise" vs
     "gold was in the right company but ranked below other passages from that
     same company". Routing can only fix the first kind.

  3. WHERE DOES THE RERANKER GO WRONG?
     Reports how often reranking moves a gold chunk out of the top 10 that
     dense retrieval had already found, which is the failure mode behind the
     -0.1021 delta.

Run:  python -u src\\eval\\diagnose_runs.py
Output: results/diagnostics_v4.md
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
    """qid -> [cid ordered by rank]"""
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


def recall_at(ranked, gold, k):
    return sum(1 for c in ranked[:k] if c in gold) / max(len(gold), 1)


def boot_mean_ci(v, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), \
        float(np.percentile(b, 97.5))


def main():
    qrels = load_qrels(PROC / "qrels_test.tsv")
    queries = {}
    for l in open(PROC / "queries_test.jsonl", encoding="utf-8"):
        r = json.loads(l)
        queries[str(r["_id"])] = r["text"]

    corpus_ticks = set()
    for l in open(PROC / "corpus.jsonl", encoding="utf-8"):
        corpus_ticks.add(json.loads(l)["_id"].split("::")[0])

    qids = [q for q in queries if q in qrels]

    named, unnamed = [], []
    for q in qids:
        hits = set(TICKRE.findall(queries[q])) & corpus_ticks
        (named if len(hits) == 1 else unnamed).append(q)

    runs = {}
    for p in sorted(RUNDIR.glob("*.trec")):
        runs[p.stem.replace("_at_", "@")] = load_trec(p)

    L = ["# Offline diagnostics on run files", "",
         f"Test questions: {len(qids)}  |  ticker named: {len(named)}  |  "
         f"not named: {len(unnamed)}", ""]

    # ---------------------------------------------- 1. circularity check
    L += ["## 1. Circularity check: is the routing gain an artifact?", "",
          "If gold labels were ticker-dependent, ft_dense (which uses NO ticker",
          "information) would score very differently on the two groups.", "",
          "| System | group | n | NDCG@10 | 95% CI |", "|---|---|---|---|---|"]

    for sysname in ("ft_dense", "small_dense", "ft_dense_ticker"):
        run = runs.get(sysname)
        if not run:
            continue
        for label, group in (("ticker named", named), ("not named", unnamed)):
            v = [ndcg_at(run.get(q, []), qrels[q]) for q in group]
            m, lo, hi = boot_mean_ci(v)
            L.append(f"| {sysname} | {label} | {len(group)} | {m:.4f} | "
                     f"[{lo:.4f}, {hi:.4f}] |")

    # ------------------------------------------- 2. what routing fixed
    L += ["", "## 2. What did routing actually fix?", "",
          "ft_dense failures split by cause. Routing can only fix the first.", ""]

    run = runs.get("ft_dense", {})
    wrong_co, right_co_wrong_chunk, found = 0, 0, 0
    frac_correct_company = []
    for q in qids:
        ranked = run.get(q, [])
        gold = qrels[q]
        gold_tick = next(iter(gold)).split("::")[0]
        top10 = ranked[:10]
        if any(c in gold for c in top10):
            found += 1
        elif any(c.split("::")[0] == gold_tick for c in top10):
            right_co_wrong_chunk += 1
        else:
            wrong_co += 1
        if top10:
            frac_correct_company.append(
                sum(1 for c in top10 if c.split("::")[0] == gold_tick) / len(top10))

    n = len(qids)
    L += [f"- gold found in top-10: **{found}** ({100 * found / n:.1f}%)",
          f"- missed, but the right company was in top-10 "
          f"(routing cannot help): **{right_co_wrong_chunk}** "
          f"({100 * right_co_wrong_chunk / n:.1f}%)",
          f"- missed, right company absent from top-10 entirely "
          f"(routing fixes this): **{wrong_co}** ({100 * wrong_co / n:.1f}%)",
          "",
          f"Mean fraction of ft_dense's top-10 that comes from the correct "
          f"company: **{np.mean(frac_correct_company):.3f}**", ""]

    # ------------------------------------------- 3. reranker failure mode
    L += ["## 3. Where does reranking go wrong?", "",
          "Gold chunks that dense retrieval placed in the top 10, and what "
          "reranking then did with them.", "",
          "| reranked system | gold kept in top-10 | gold pushed out | "
          "gold pulled in |", "|---|---|---|---|"]

    base = runs.get("ft_dense", {})
    for sysname in ("ft_dense_rerank@50", "ft_dense_rerank_base@50",
                    "ft_dense_ticker_rerank@50"):
        rr = runs.get(sysname)
        if not rr:
            continue
        ref = runs.get("ft_dense_ticker", {}) if "ticker" in sysname else base
        kept = pushed = pulled = 0
        for q in qids:
            gold = qrels[q]
            b10 = {c for c in ref.get(q, [])[:10] if c in gold}
            r10 = {c for c in rr.get(q, [])[:10] if c in gold}
            kept += len(b10 & r10)
            pushed += len(b10 - r10)
            pulled += len(r10 - b10)
        L.append(f"| {sysname} | {kept} | {pushed} | {pulled} |")

    # -------------------------------- 4. routed vs unrouted on named only
    L += ["", "## 4. Routing effect measured only on the queries it applies to",
          "",
          "The headline +0.3468 is diluted by 235 unnamed queries where routing",
          "falls back to plain dense retrieval. Restricted to the 853 queries",
          "that actually get routed:", "",
          "| System | NDCG@10 | Recall@100 | 95% CI (NDCG) |", "|---|---|---|---|"]

    for sysname in ("ft_dense", "ft_dense_ticker", "ft_dense_ticker_rerank@50"):
        r = runs.get(sysname)
        if not r:
            continue
        v = [ndcg_at(r.get(q, []), qrels[q]) for q in named]
        rc = [recall_at(r.get(q, []), qrels[q], 100) for q in named]
        m, lo, hi = boot_mean_ci(v)
        L.append(f"| {sysname} | {m:.4f} | {np.mean(rc):.4f} | "
                 f"[{lo:.4f}, {hi:.4f}] |")

    out = "\n".join(L) + "\n"
    print(out)
    (RESULTS / "diagnostics_v4.md").write_text(out, encoding="utf-8")
    print("saved -> results/diagnostics_v4.md")


if __name__ == "__main__":
    main()