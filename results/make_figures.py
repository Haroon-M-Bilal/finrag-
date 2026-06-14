"""
Make paper-ready figures from results JSONs.
Outputs 300-dpi PNGs to results/figures/ :
    fig1_main_comparison.png   - best baseline vs best system (bar)
    fig2_recall_curve.png      - recall@k baseline vs ours (line)
    fig3_ablation.png          - all systems on MRR@10 (the ladder)
    fig4_generation.png        - base vs QLoRA (bar)
    fig5_pipeline.png          - system diagram

Run:  python results\\make_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = Path("results"); FIG = R / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})
C_BASE, C_OURS = "#9aa0a6", "#c2410c"   # grey baseline, orange ours

comp = json.load(open(R / "comparison.json"))
gen = json.load(open(R / "generation.json")) if (R / "generation.json").exists() else None
BASE_BEST = "hybrid_rerank"      # best baseline configuration
OURS_BEST = "ft_dense"           # best fine-tuned configuration


def fig1():
    metrics = ["mrr@10", "ndcg@10", "recall@10"]
    labels = ["MRR@10", "NDCG@10", "Recall@10"]
    b = [comp[BASE_BEST][m] for m in metrics]
    o = [comp[OURS_BEST][m] for m in metrics]
    x = range(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w/2 for i in x], b, w, label="Baseline (off-the-shelf)", color=C_BASE)
    ax.bar([i + w/2 for i in x], o, w, label="Ours (fine-tuned)", color=C_OURS)
    for i in x:
        ax.text(i + w/2, o[i] + 0.008, f"{o[i]/b[i]:.1f}\u00d7", ha="center",
                fontweight="bold", color=C_OURS)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_title("Retrieval quality: baseline vs fine-tuned")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig1_main_comparison.png"); plt.close()


def fig2():
    ks = [5, 10, 20]
    b = [comp[BASE_BEST][f"recall@{k}"] for k in ks]
    o = [comp[OURS_BEST][f"recall@{k}"] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, b, "o--", color=C_BASE, label="Baseline", linewidth=2, markersize=7)
    ax.plot(ks, o, "o-", color=C_OURS, label="Ours", linewidth=2, markersize=7)
    ax.set_xticks(ks); ax.set_xlabel("k"); ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k: baseline vs fine-tuned"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig2_recall_curve.png"); plt.close()


def fig3():
    order = ["dense_only", "hybrid", "hybrid_rerank", "ft_hybrid_rerank", "ft_hybrid", "ft_dense"]
    names = ["Dense", "Hybrid", "Hybrid+Rerank", "FT Hybrid+Rerank", "FT Hybrid", "FT Dense (ours)"]
    vals = [comp[k]["mrr@10"] for k in order]
    colors = [C_BASE]*3 + [C_OURS]*3
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.004, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center")
    ax.set_xlabel("MRR@10"); ax.set_title("Ablation: contribution of each component")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(FIG / "fig3_ablation.png"); plt.close()


def fig4():
    if not gen:
        return
    metrics = ["ROUGE-L", "BERTScore-F1"]
    b = [gen["base"][m] for m in metrics]
    o = [gen["qlora"][m] for m in metrics]
    x = range(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([i - w/2 for i in x], b, w, label="Base Qwen2.5-7B", color=C_BASE)
    ax.bar([i + w/2 for i in x], o, w, label="+ QLoRA (ours)", color=C_OURS)
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score"); ax.set_title("Generation quality: base vs QLoRA")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig4_generation.png"); plt.close()


def fig5():
    fig, ax = plt.subplots(figsize=(9, 2.4)); ax.axis("off")
    steps = ["Question", "Fine-tuned\nEmbedder", "BM25 +\nRRF fusion",
             "Fine-tuned\nReranker", "QLoRA\nQwen-7B", "Answer"]
    x = 0
    centers = []
    for i, s in enumerate(steps):
        w = 1.5
        box = FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0.05",
                             fc="#fff4ee" if 0 < i < 5 else "#eef2f7",
                             ec=C_OURS if 0 < i < 5 else "#5b6770", lw=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, 0.5, s, ha="center", va="center", fontsize=10)
        centers.append((x + w, x))
        x += w + 0.7
    for i in range(len(steps) - 1):
        ax.add_patch(FancyArrowPatch((centers[i][0], 0.5), (centers[i+1][1], 0.5),
                     arrowstyle="-|>", mutation_scale=14, color="#5b6770"))
    ax.set_xlim(-0.3, x); ax.set_ylim(-0.3, 1.3)
    ax.set_title("FinRAG pipeline (orange = fine-tuned on FinDER)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig5_pipeline.png"); plt.close()


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("DONE. figures saved to results/figures/")
    for p in sorted(FIG.glob("*.png")):
        print(" ", p)