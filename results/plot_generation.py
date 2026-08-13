"""
Plot generation-quality results (base vs QLoRA) for every evaluated model.

Reads results/generation_all.json (written by run_generation_eval.py) and writes:
    results/figures/fig4_generation.png        - base vs QLoRA for the main model
    results/figures/fig8_model_comparison.png  - all models side by side (if >1)
    results/figures/fig9_improvement.png       - relative gain per model (if >1)

Run:  python results\\plot_generation.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results"); FIG = R / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})
C_BASE, C_FT = "#9aa0a6", "#c2410c"

src = R / "generation_all.json"
if not src.exists():
    raise SystemExit("results/generation_all.json not found. Run run_generation_eval.py first.")
allr = json.load(open(src))
names = list(allr.keys())
short = [n.split("/")[-1].replace("-Instruct", "") for n in names]
print("models found:", short)

# ---------- Fig 4: base vs QLoRA (main model = 7B if present) ----------
main = next((n for n in names if "7B" in n and "Mistral" not in n), names[0])
d = allr[main]
metrics = ["ROUGE-L", "BERTScore-F1"]
b = [d["base"][m] for m in metrics]; f = [d["qlora"][m] for m in metrics]
x = np.arange(len(metrics)); w = 0.38
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.bar(x - w/2, b, w, label="Base", color=C_BASE)
ax.bar(x + w/2, f, w, label="+ QLoRA (ours)", color=C_FT)
for i in x:
    ax.text(i + w/2, f[i] + 0.008, f"{f[i]:.3f}", ha="center", fontsize=9,
            fontweight="bold", color=C_FT)
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylabel("Score")
ax.set_title(f"Generation quality: {main.split('/')[-1]}")
ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig4_generation.png"); plt.close()
print("saved -> fig4_generation.png")

# ---------- Fig 8 + 9: across models ----------
if len(names) > 1:
    # grouped bars: ROUGE-L base vs qlora per model
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, m in zip(axes, ["ROUGE-L", "BERTScore-F1"]):
        bb = [allr[n]["base"][m] for n in names]
        ff = [allr[n]["qlora"][m] for n in names]
        x = np.arange(len(names)); w = 0.38
        ax.bar(x - w/2, bb, w, label="Base", color=C_BASE)
        ax.bar(x + w/2, ff, w, label="+ QLoRA", color=C_FT)
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
        ax.set_title(m); ax.set_ylabel("Score")
    axes[0].legend(fontsize=9)
    fig.suptitle("Generation quality across generator models")
    fig.tight_layout(); fig.savefig(FIG / "fig8_model_comparison.png"); plt.close()
    print("saved -> fig8_model_comparison.png")

    # relative improvement in ROUGE-L
    gains = [allr[n]["qlora"]["ROUGE-L"] / max(allr[n]["base"]["ROUGE-L"], 1e-9)
             for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(short, gains, color=C_FT)
    for bar, g in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width()/2, g + 0.05, f"{g:.1f}x",
                ha="center", fontweight="bold", fontsize=9)
    ax.axhline(1.0, color="#5b6770", linestyle="--", linewidth=1)
    ax.set_ylabel("ROUGE-L gain over base"); ax.set_title("Relative gain from QLoRA fine-tuning")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig9_improvement.png"); plt.close()
    print("saved -> fig9_improvement.png")

# ---------- print the table ----------
cols = ["ROUGE-L", "BLEU", "BERTScore-F1"]
print("\n| Model | System | " + " | ".join(cols) + " |")
print("|" + "---|" * (len(cols) + 2))
for n, s in zip(names, short):
    for tag, key in (("base", "base"), ("+ QLoRA", "qlora")):
        print(f"| {s} | {tag} | " + " | ".join(f"{allr[n][key][c]:.4f}" for c in cols) + " |")