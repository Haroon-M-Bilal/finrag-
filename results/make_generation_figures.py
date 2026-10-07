"""
Generation figures: three models, two context conditions.

Built from results/generation_*_v4.json so the figures cannot drift from the
tables. Same grayscale-safe style as the retrieval figures: series are
distinguished by shade and hatch, not colour alone.

Run:  python -u results/make_generation_figures.py
Output: results/figures/fig_v4_gen_*.png (300 dpi) and .pdf (vector)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight",
})

GREY = ["#2b2b2b", "#5a5a5a", "#8a8a8a", "#b4b4b4", "#d6d6d6"]

MODELS = [
    ("qwen2-5-3b-instruct", "Qwen2.5-3B"),
    ("llama-3-2-3b-instruct", "Llama-3.2-3B"),
    ("mistral-7b-instruct-v0-3", "Mistral-7B"),
]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_v4_gen_{name}.{ext}")
    plt.close(fig)
    print(f"  wrote fig_v4_gen_{name}.png / .pdf")


def main():
    data = {}
    for tag, label in MODELS:
        p = RESULTS / f"generation_{tag}_v4.json"
        if not p.exists():
            print(f"  missing {p}, skipping {label}")
            continue
        data[label] = json.load(open(p))["report"]
    if not data:
        raise SystemExit("no generation results found")
    labels = list(data)
    print(f"loaded {len(labels)} models: {', '.join(labels)}")

    # ------------------------------------------- FIG 1: base vs fine-tuned
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(labels))
    w = 0.35
    base = [data[m]["gold_base"]["ROUGE-L"] for m in labels]
    ft = [data[m]["gold_ft"]["ROUGE-L"] for m in labels]

    ax.bar(x - w / 2, base, w, label="base", color=GREY[3],
           edgecolor="#000000", lw=0.6)
    ax.bar(x + w / 2, ft, w, label="+ QLoRA", color=GREY[0],
           edgecolor="#000000", lw=0.6)
    for i, (b, f) in enumerate(zip(base, ft)):
        ax.text(i - w / 2, b + 0.006, f"{b:.3f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, f + 0.006, f"{f:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("ROUGE-L")
    ax.set_ylim(0, max(ft) * 1.25)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Gold context (generation ceiling)")
    save(fig, "finetuning")

    # ------------------------------ FIG 2: gold vs retrieved, fine-tuned only
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    gold = [data[m]["gold_ft"]["ROUGE-L"] for m in labels]
    retr = [data[m]["retrieved_ft"]["ROUGE-L"] for m in labels]
    ax.bar(x - w / 2, gold, w, label="gold context", color=GREY[1],
           edgecolor="#000000", lw=0.6)
    ax.bar(x + w / 2, retr, w, label="retrieved (top-3)", color=GREY[3],
           hatch="///", edgecolor="#000000", lw=0.6)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("ROUGE-L")
    ax.set_ylim(0, max(gold) * 1.25)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Retrieval-attributable gap")
    save(fig, "gold_vs_retrieved")

    # ------------------------------------------ FIG 3: numeric exact-match
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ng = [data[m]["gold_ft"]["numeric_em"] for m in labels]
    nr = [data[m]["retrieved_ft"]["numeric_em"] for m in labels]
    n_num = data[labels[0]]["gold_ft"]["numeric_n"]
    ax.bar(x - w / 2, ng, w, label="gold context", color=GREY[1],
           edgecolor="#000000", lw=0.6)
    ax.bar(x + w / 2, nr, w, label="retrieved", color=GREY[3],
           hatch="///", edgecolor="#000000", lw=0.6)
    for i, (a, b) in enumerate(zip(ng, nr)):
        ax.text(i - w / 2, a + 0.006, f"{a:.3f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, b + 0.006, f"{b:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("Numeric exact-match")
    ax.set_ylim(0, max(ng) * 1.35)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Financial calculation (n={n_num})")
    save(fig, "numeric")

    # -------------------------------- FIG 4: numeric outcome decomposition
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    keys = [("correct", "correct", GREY[0], ""),
            ("scale_error", "scale error", GREY[2], "..."),
            ("wrong", "wrong value", GREY[3], "///"),
            ("abstained", "abstained", GREY[4], "xxx"),
            ("malformed", "malformed", "#ffffff", "\\\\\\")]
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for k, lab, c, h in keys:
        vals = np.array([data[m]["gold_ft"]["numeric_outcomes"][k]
                         for m in labels], dtype=float)
        ax.barh(y, vals, left=left, color=c, hatch=h, edgecolor="#000000",
                lw=0.6, height=0.55, label=lab)
        left += vals
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"questions (n={int(left[0])}, gold context)")
    ax.legend(frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Where numeric answers fail")
    save(fig, "numeric_breakdown")

    print(f"\nfigures -> {FIGDIR}")


if __name__ == "__main__":
    main()