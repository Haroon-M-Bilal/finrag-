"""
Pipeline architecture figure for the paper.

Three lanes: offline corpus construction, offline label construction, and the
online query path. Grayscale-safe, sized for a two-column IEEE layout at full
width.

LAYOUT NOTE: every box shares one width and one horizontal pitch, set by the
constants below. The first version positioned boxes individually, which let
neighbouring boxes overlap and text bleed across them.

Run:  python -u results/make_pipeline_figure.py
Output: results/figures/fig_v4_pipeline.png (300 dpi) and .pdf (vector)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGDIR = Path("results/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 7, "figure.dpi": 300,
                     "savefig.bbox": "tight"})

DARK, MID, LIGHT, PALE = "#2b2b2b", "#8a8a8a", "#dcdcdc", "#f4f4f4"

X0 = 0.055                 # left edge of every lane
BW, BH = 0.158, 0.105      # box width / height
GAP = 0.030                # horizontal gap between boxes
STEP = BW + GAP


def box(ax, x, y, w, h, text, face=PALE, bold=False, fs=6.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=face, edgecolor=DARK, linewidth=0.9, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="white" if face == DARK else "black",
            fontweight="bold" if bold else "normal", linespacing=1.4,
            zorder=4)


def arrow(ax, x1, y1, x2, y2, ls="-", color=DARK, lw=0.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=lw, linestyle=ls,
                                 color=color, shrinkA=0, shrinkB=0, zorder=2))


def lane(ax, y, text):
    ax.text(0.012, y, text, ha="center", va="center", fontsize=7.5,
            fontweight="bold", rotation=90, color=DARK)


def note(ax, y, text):
    ax.text(X0, y, text, fontsize=6.1, style="italic", color="#484848")


def row(ax, y, items, n=5):
    xs = [X0 + i * STEP for i in range(n)]
    for x, (lab, f, b) in zip(xs, items):
        box(ax, x, y, BW, BH, lab, face=f, bold=b)
    for i in range(n - 1):
        arrow(ax, xs[i] + BW, y + BH / 2, xs[i + 1], y + BH / 2)
    return xs


def main():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ------------------------------------------------ lane A: corpus (offline)
    yA = 0.845
    note(ax, yA + BH + 0.022,
         "Chunk size is set in the ENCODER's tokenizer, not in words: "
         "financial text runs 1.6\u20131.7 tokens/word, tables far higher.")
    xs = row(ax, yA, [
        ("497 S&P 500\n10-K filings", LIGHT, False),
        ("Text\nextraction", PALE, False),
        ("Token chunking\n448 / 112\nBGE tokenizer", PALE, False),
        ("181,867 chunks\nmax 454 tok\n0 over limit", PALE, False),
        ("Fine-tuned\nembedder\n\u2192 FAISS", MID, False),
    ])
    lane(ax, yA + BH / 2, "A. Corpus")

    # ------------------------------------------------- lane B: labels (offline)
    yB = 0.555
    note(ax, yB + BH + 0.022,
         "Gold is restricted to one filing: 52.8% of questions previously had "
         "gold in multiple filings (max 38) through 10-K boilerplate.")
    row(ax, yB, [
        ("FinDER\n5,703 questions\n+ references", LIGHT, False),
        ("Resolve filing\nticker, else\nTF-IDF", PALE, False),
        ("Union coverage\n\u2265 0.80 within\nthat filing", PALE, False),
        ("Graded qrels\n5,303 questions\n2.18 chunks/q", PALE, False),
        ("Frozen split\n392 / 98 filings\n4,215 / 1,088 q", MID, False),
    ])
    lane(ax, yB + BH / 2, "B. Labels")

    # dashed feeds
    xr = xs[4] + BW / 2
    arrow(ax, xr, yA, xr, yB + BH, ls=(0, (3, 2)), color=MID)
    arrow(ax, xr, yB, xr, 0.325, ls=(0, (3, 2)), color=MID)

    # -------------------------------------------------- lane C: query (online)
    yC = 0.215
    note(ax, yC + BH + 0.022,
         "Document routing is a string match \u2014 no LLM, no API. Only 14.4% "
         "of unrouted top-10 results come from the correct company.")
    row(ax, yC, [
        ("Query\n(names CBOE)", LIGHT, False),
        ("Ticker regex\n1 match \u2192 route\n78.4% of queries", DARK, True),
        ("Candidate pool\nrouted: 1 filing\nelse: top-200", DARK, True),
        ("Cross-encoder\nrerank top-50\nclip 460 XLM-R", DARK, True),
        ("Top-3 chunks\nany-gold 0.788\n\u2192 QLoRA gen", MID, False),
    ])
    lane(ax, yC + BH / 2, "C. Query")

    # generator output, placed to the right of the results strip
    box(ax, X0 + 4 * STEP, 0.045, BW, BH,
        "Prose answer\n+ ANSWER: block\n111.5 | million | USD", face=PALE)
    arrow(ax, X0 + 4 * STEP + BW / 2, yC,
          X0 + 4 * STEP + BW / 2, 0.045 + BH)

    # ------------------------------------------------------------ result strip
    ax.text(X0, 0.152, "NDCG@10, 853 routed test queries",
            fontsize=7, fontweight="bold")
    rw = 0.175
    for i, (lab, val, f, b) in enumerate([
            ("fine-tuned dense", "0.157", LIGHT, False),
            ("+ routing", "0.600", MID, False),
            ("+ reranking", "0.660", DARK, True)]):
        x = X0 + i * (rw + 0.026)
        box(ax, x, 0.060, rw, 0.070, f"{lab}\n{val}", face=f, bold=b, fs=7)
        if i:
            arrow(ax, x - 0.026, 0.095, x, 0.095)

    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_v4_pipeline.{ext}")
    plt.close(fig)
    print(f"wrote {FIGDIR / 'fig_v4_pipeline.png'} and .pdf")


if __name__ == "__main__":
    main()