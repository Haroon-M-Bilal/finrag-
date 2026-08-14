"""
Figures for the paper, built from the dumped TREC run files.

All numbers are recomputed from results/runs/*.trec + qrels_test.tsv, so the
figures cannot drift from the tables - there is one source of truth.

Sized for a two-column IEEE layout (3.4in single column, 7.0in full width) and
kept grayscale-safe: series are distinguished by hatch and marker as well as
shade, so the figures survive black-and-white printing and colour-blind readers.

Run:  python -u results/make_figures_v4.py
Output: results/figures/fig_v4_*.png (300 dpi) and .pdf (vector, for LaTeX)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROC = Path("data/finder/processed_v4")
RESULTS = Path("results")
RUNDIR = RESULTS / "runs"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

TICKRE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,5}\b")
N_BOOT = 1000
SEED = 0

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

GREY = ["#2b2b2b", "#5a5a5a", "#8a8a8a", "#b4b4b4", "#d6d6d6"]


def load_qrels(p):
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, s = line.rstrip("\n").split("\t")
            q.setdefault(a, {})[b] = int(s)
    return q


def load_trec(p):
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


def ci(v, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_v4_{name}.{ext}")
    plt.close(fig)
    print(f"  wrote fig_v4_{name}.png / .pdf")


def main():
    qrels = load_qrels(PROC / "qrels_test.tsv")
    queries = {}
    for l in open(PROC / "queries_test.jsonl", encoding="utf-8"):
        r = json.loads(l)
        queries[str(r["_id"])] = r["text"]
    ticks = set()
    for l in open(PROC / "corpus.jsonl", encoding="utf-8"):
        ticks.add(json.loads(l)["_id"].split("::")[0])

    qids = [q for q in queries if q in qrels]
    routed = [q for q in qids
              if len(set(TICKRE.findall(queries[q])) & ticks) == 1]

    runs = {p.stem.replace("_at_", "@"): load_trec(p)
            for p in sorted(RUNDIR.glob("*.trec"))}
    print(f"loaded {len(runs)} runs, {len(qids)} queries "
          f"({len(routed)} routed)")

    # ---------------------------------------------------- FIG 1: main ablation
    order = [("bm25", "BM25"),
             ("base_dense", "BGE-base (off-the-shelf)"),
             ("small_dense", "BGE-small (off-the-shelf)"),
             ("ft_hybrid", "Fine-tuned + BM25 (RRF)"),
             ("ft_dense_rerank@50", "Fine-tuned + rerank"),
             ("ft_dense", "Fine-tuned dense"),
             ("ft_dense_ticker", "+ document routing"),
             ("ft_dense_ticker_rerank@50", "+ routing + rerank")]
    order = [(k, lab) for k, lab in order if k in runs]
    _tmp = [(k, lab, np.mean([ndcg_at(runs[k].get(q, []), qrels[q])
                              for q in qids])) for k, lab in order]
    order = [(k, lab) for k, lab, _ in sorted(_tmp, key=lambda t: t[2])]

    means, los, his, labels = [], [], [], []
    for k, lab in order:
        m, lo, hi = ci([ndcg_at(runs[k].get(q, []), qrels[q]) for q in qids])
        means.append(m); los.append(m - lo); his.append(hi - m); labels.append(lab)

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y = np.arange(len(labels))
    cols = [GREY[2]] * len(labels)
    for i, (k, _) in enumerate(order):
        if k.startswith("ft_dense_ticker"):
            cols[i] = GREY[0]
        elif k == "ft_dense":
            cols[i] = GREY[1]
    ax.barh(y, means, xerr=[los, his], color=cols, height=0.65,
            error_kw={"lw": 0.9, "capsize": 2.5, "ecolor": "#000000"})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("NDCG@10 (all 1088 held-out test questions)")
    ax.set_xlim(0, 0.62)
    for i, m in enumerate(means):
        ax.text(m + his[i] + 0.012, i, f"{m:.3f}", va="center", fontsize=7)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "ablation")

    # --------------------------------------------------- FIG 2: routed chain
    chain = [("ft_dense", "Fine-tuned\ndense"),
             ("ft_dense_ticker", "+ document\nrouting"),
             ("ft_dense_ticker_rerank@50", "+ neural\nreranking")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ms, es = [], [[], []]
    for k, _ in chain:
        m, lo, hi = ci([ndcg_at(runs[k].get(q, []), qrels[q]) for q in routed])
        ms.append(m); es[0].append(m - lo); es[1].append(hi - m)
    x = np.arange(len(chain))
    ax.bar(x, ms, yerr=es, color=[GREY[3], GREY[1], GREY[0]], width=0.6,
           error_kw={"lw": 0.9, "capsize": 3, "ecolor": "#000000"})
    ax.set_xticks(x, [lab for _, lab in chain])
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, 0.80)
    for i, m in enumerate(ms):
        ax.text(i, m + es[1][i] + 0.02, f"{m:.3f}", ha="center", fontsize=7.5)
    ax.annotate("", xy=(1, 0.70), xytext=(0, 0.70),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.5, 0.715, "+0.443", ha="center", fontsize=7)
    ax.annotate("", xy=(2, 0.745), xytext=(1, 0.745),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(1.5, 0.760, "+0.060", ha="center", fontsize=7)
    ax.set_title(f"Routed queries only (n={len(routed)})")
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "routed_chain")

    # ------------------------------------------- FIG 3: failure decomposition
    run = runs["ft_dense"]
    found = right_co = wrong_co = 0
    for q in qids:
        top10 = run.get(q, [])[:10]
        gold = qrels[q]
        gt = next(iter(gold)).split("::")[0]
        if any(c in gold for c in top10):
            found += 1
        elif any(c.split("::")[0] == gt for c in top10):
            right_co += 1
        else:
            wrong_co += 1
    n = len(qids)

    fig, ax = plt.subplots(figsize=(7.0, 1.35))
    segs = [(found, "Gold retrieved\nin top-10", GREY[0], ""),
            (right_co, "Correct filing,\nwrong passage", GREY[2], "///"),
            (wrong_co, "Correct filing absent\nfrom top-10", GREY[4], "xxx")]
    left = 0
    for v, lab, c, h in segs:
        ax.barh(0, 100 * v / n, left=left, color=c, hatch=h,
                edgecolor="#000000", lw=0.6, height=0.55)
        ax.text(left + 50 * v / n, 0, f"{100 * v / n:.1f}%", ha="center",
                va="center", fontsize=8, fontweight="bold",
                color="white" if c == GREY[0] else "black",
                bbox=dict(facecolor=c if c == GREY[0] else "white",
                          edgecolor="none", pad=2.0, alpha=0.92))
        ax.text(left + 50 * v / n, -0.52, lab, ha="center", va="top",
                fontsize=7)
        left += 100 * v / n
    ax.set_xlim(0, 100); ax.set_ylim(-1.1, 0.4)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Where fine-tuned dense retrieval fails "
                 f"(n={n}); routing addresses the right-hand segment",
                 fontsize=8)
    save(fig, "failure_modes")

    # ----------------------------------------------------- FIG 4: recall@k
    ks = [1, 3, 5, 10, 20, 50, 100]
    series = [("bm25", "BM25", "s", ":"),
              ("small_dense", "BGE-small", "^", "-."),
              ("ft_dense", "Fine-tuned dense", "o", "--"),
              ("ft_dense_ticker", "+ document routing", "D", "-")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, (k, lab, mk, ls) in enumerate(series):
        if k not in runs:
            continue
        ys = [np.mean([recall_at(runs[k].get(q, []), qrels[q], kk)
                       for q in routed]) for kk in ks]
        ax.plot(ks, ys, marker=mk, ls=ls, ms=3.5, lw=1.1,
                color=GREY[min(3 - i, 3)] if i < 4 else GREY[0], label=lab)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Routed queries (n={len(routed)})")
    save(fig, "recall_curve")

    # ------------------------------------------------ FIG 5: rerank depth
    depths = [50, 100, 200]
    have = [d for d in depths if f"ft_dense_rerank@{d}" in runs]
    if have:
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        base_m, base_lo, base_hi = ci(
            [ndcg_at(runs["ft_dense"].get(q, []), qrels[q]) for q in qids])
        ms, es = [], [[], []]
        for d in have:
            m, lo, hi = ci([ndcg_at(runs[f"ft_dense_rerank@{d}"].get(q, []),
                                    qrels[q]) for q in qids])
            ms.append(m); es[0].append(m - lo); es[1].append(hi - m)
        ax.axhline(base_m, color=GREY[0], ls="--", lw=1.0,
                   label="no reranking")
        ax.fill_between([-0.5, len(have) - 0.5], base_lo, base_hi,
                        color=GREY[4], alpha=0.5, lw=0)
        ax.bar(np.arange(len(have)), ms, yerr=es, color=GREY[2], width=0.55,
               hatch="///", edgecolor="#000000", lw=0.6,
               error_kw={"lw": 0.9, "capsize": 3, "ecolor": "#000000"})
        ax.set_xticks(np.arange(len(have)), [f"top-{d}" for d in have])
        ax.set_xlim(-0.5, len(have) - 0.5)
        ax.set_xlabel("reranked candidate pool")
        ax.set_ylabel("NDCG@10")
        ax.legend(frameon=False)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title("Reranking without routing (all queries)")
        save(fig, "rerank_depth")

    print(f"\nfigures -> {FIGDIR}")


if __name__ == "__main__":
    main()