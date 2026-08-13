"""
Plot the NEW QLoRA loss curves (training + validation).

Reads results/qlora_loss_<tag>.json produced by finetune_qlora.py and writes
results/figures/fig2_qlora_loss.png  (replaces the old noisy Fig. 2)

If several models have been trained, it also writes
results/figures/fig7_val_loss_all_models.png comparing their validation curves.

Run:  python results\\plot_loss.py
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results"); FIG = R / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})
C_TRAIN, C_VAL = "#9aa0a6", "#c2410c"

files = sorted(glob.glob(str(R / "qlora_loss_*.json")))
if not files:
    raise SystemExit("No qlora_loss_*.json found in results/. Train a model first.")
print("found:", [Path(f).name for f in files])

# ---- Fig 2: main model, train vs validation ----
main = None
for f in files:
    if "7b" in Path(f).name and "mistral" not in Path(f).name:
        main = f; break
main = main or files[0]
d = json.load(open(main))
tr = d["train"]; va = d["val"]

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot([p[0] for p in tr], [p[1] for p in tr], color=C_TRAIN, linewidth=1.2,
        alpha=0.85, label="Training loss (per step)")
ax.plot([p[0] for p in va], [p[1] for p in va], color=C_VAL, linewidth=2.2,
        marker="o", markersize=4, label="Validation loss")
ax.set_xlabel("Training step"); ax.set_ylabel("Loss")
ax.set_title(f"QLoRA fine-tuning: {d['model'].split('/')[-1]}")
ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig2_qlora_loss.png"); plt.close()
print("saved -> results/figures/fig2_qlora_loss.png")
print(f"  validation loss: {va[0][1]:.4f} -> {va[-1][1]:.4f}")

# ---- Fig 7: validation curves across all trained models ----
if len(files) > 1:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = ["#c2410c", "#1d4ed8", "#15803d", "#7c3aed", "#b45309"]
    for i, f in enumerate(files):
        dd = json.load(open(f))
        v = dd["val"]
        ax.plot([p[0] for p in v], [p[1] for p in v], linewidth=2,
                marker="o", markersize=3, color=colors[i % len(colors)],
                label=dd["model"].split("/")[-1])
    ax.set_xlabel("Training step"); ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss across generator models")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig7_val_loss_all_models.png"); plt.close()
    print("saved -> results/figures/fig7_val_loss_all_models.png")