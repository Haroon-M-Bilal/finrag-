"""
Pipeline architecture figure for the paper.

Three lanes: offline corpus construction, offline label construction, and the
online query path. Grayscale-safe, sized for a two-column IEEE layout at full
width.

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

plt.rcParams.update({
    "font.size": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

DARK = "#2b2b2b"
MID = "#7a7a7a"
LIGHT = "#dcdcdc"
PALE = "#f2f2f2"


def box(ax, x, y, w, h, text, face=PALE, edge=DARK, bold=False, fs=7):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="white" if face == DARK else "black",
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", lw=0.9, color=DARK):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8,
        linewidth=lw, linestyle=ls, color=color,
        shrinkA=0, shrinkB=0))


def lane_label(ax, y, text):
    ax.text(-0.015, y, text, ha="right", va="center", fontsize=7.5,
            fontweight="bold", rotation=90, color=DARK)


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    W, H = 0.145, 0.088
    gap = 0.028

    # ------------------------------------------------ lane A: corpus (offline)
    yA = 0.845
    xs = [0.02 + i * (W + gap) for i in range(5)]
    labels = [
        "497 S&P 500\n10-K filings\n(HTML)",
        "Text extraction\n(BeautifulSoup)",
        "Token chunking\n448 tok / 112 overlap\nBGE tokenizer",
        "181,867 chunks\nmax 454 tok\n0 over limit",
        "Fine-tuned embedder\nbge-small-finder-v4\n\u2192 FAISS IndexFlatIP",
    ]
    faces = [LIGHT, PALE, PALE, PALE, MID]
    for x, lab, f in zip(xs, labels, faces):
        box(ax, x, yA, W, H, lab, face=f)
    for i in range(4):
        arrow(ax, xs[i] + W, yA + H / 2, xs[i + 1], yA + H / 2)
    lane_label(ax, yA + H / 2, "A. Corpus")

    ax.text(0.02, yA + H + 0.028,
            "Chunk size is set in the ENCODER's tokenizer, not in words: "
            "financial text runs 1.6\u20131.7 tokens/word and number-dense "
            "tables far higher.",
            fontsize=6.4, style="italic", color="#444444")

    # ------------------------------------------------- lane B: labels (offline)
    yB = 0.545
    labels = [
        "FinDER\n5,703 questions\n+ reference text",
        "Resolve filing\nticker in query,\nelse TF-IDF",
        "Union coverage\n\u2265 0.80 WITHIN\nthat filing only",
        "Graded qrels\n5,303 questions\n2.18 chunks/query",
        "Frozen split\n392 / 98 filings\n4,215 / 1,088 q",
    ]
    faces = [LIGHT, PALE, PALE, PALE, MID]
    for x, lab, f in zip(xs, labels, faces):
        box(ax, x, yB, W, H, lab, face=f)
    for i in range(4):
        arrow(ax, xs[i] + W, yB + H / 2, xs[i + 1], yB + H / 2)
    lane_label(ax, yB + H / 2, "B. Labels")

    ax.text(0.02, yB + H + 0.028,
            "Restricting gold to one filing removes cross-company boilerplate "
            "matches: 52.8% of questions previously had gold in multiple "
            "filings (max 38).",
            fontsize=6.4, style="italic", color="#444444")

    # index + split feed the online path
    arrow(ax, xs[4] + W / 2, yA, xs[4] + W / 2, yB + H, ls=(0, (3, 2)),
          color=MID)
    arrow(ax, xs[4] + W / 2, yB, xs[4] + W / 2, 0.375, ls=(0, (3, 2)),
          color=MID)

    # -------------------------------------------------- lane C: query (online)
    yC = 0.20
    cx = [0.02, 0.185, 0.35, 0.545, 0.71, 0.855]
    cw = [0.135, 0.135, 0.165, 0.135, 0.115, 0.125]
    labels = [
        "Query\n\"Delta in CBOE Data\n& Access rev\"",
        "Ticker regex\n1 match \u2192 route\n(78.4% of queries)",
        "Candidate pool\nrouted: that filing only\nelse: FAISS top-200",
        "Cross-encoder rerank\ntop-50, clipped to 460\nXLM-R tokens",
        "Top-3 chunks\nany-gold 0.788",
        "QLoRA generator\nQwen2.5-3B / 7B\n4-bit",
    ]
    faces = [LIGHT, DARK, DARK, DARK, PALE, MID]
    for x, w, lab, f in zip(cx, cw, labels, faces):
        box(ax, x, yC, w, H, lab, face=f, bold=(f == DARK), fs=6.6)
    for i in range(5):
        arrow(ax, cx[i] + cw[i], yC + H / 2, cx[i + 1], yC + H / 2)
    lane_label(ax, yC + H / 2, "C. Query")

    box(ax, 0.855, yC - 0.155, 0.125, H,
        "Prose answer\n+ ANSWER: block\n111.5 | million | USD",
        face=PALE, fs=6.6)
    arrow(ax, 0.855 + 0.125 / 2, yC, 0.855 + 0.125 / 2, yC - 0.155 + H)

    ax.text(0.02, yC + H + 0.028,
            "Document routing is a string match \u2014 no LLM, no API. Only "
            "14.4% of unrouted top-10 results come from the correct company.",
            fontsize=6.4, style="italic", color="#444444")

    # ------------------------------------------------------------ result strip
    ax.text(0.02, 0.055,
            "NDCG@10 on the 853 routed test queries:", fontsize=7,
            fontweight="bold")
    for i, (lab, val, f) in enumerate([
            ("fine-tuned dense", "0.157", LIGHT),
            ("+ document routing", "0.600", MID),
            ("+ reranking", "0.660", DARK)]):
        x = 0.30 + i * 0.235
        box(ax, x, 0.005, 0.215, 0.058, f"{lab}   {val}", face=f, fs=6.8,
            bold=(f == DARK))
        if i:
            arrow(ax, x - 0.02, 0.034, x, 0.034)

    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_v4_pipeline.{ext}")
    plt.close(fig)
    print(f"wrote {FIGDIR / 'fig_v4_pipeline.png'} and .pdf")


if __name__ == "__main__":
    main()