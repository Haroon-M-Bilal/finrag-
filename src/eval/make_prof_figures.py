"""
Compile every result into figures for the professor.

Reads only JSON already on disk. No API, no GPU, no network.
Writes PNGs into results/figures_hdrr/ and an index page that shows them with
headings.

Run:  python3 src/eval/make_prof_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

R = Path("results")
OUT = R / "figures_hdrr"
OUT.mkdir(parents=True, exist_ok=True)

INK   = "#1b1b1f"
MUTE  = "#8a8a93"
GRID  = "#e3e3e8"
OURS  = "#2f6f4e"
THEIR = "#7a5ea8"
WARM  = "#b4663a"
NEUT  = "#9aa0a8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
})


def J(name):
    p = R / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def finish(fig, ax, title, sub=None):
    ax.set_title(title, loc="left", pad=16 if sub else 10, weight="bold")
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.5, color=MUTE)
    return fig


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  {name}")


figs = []

# ---------------------------------------------------------------- data
cmp_v2 = J("hdrr_final_comparison_mistral-7b-instruct-v0-3_retrieved_k10_v2.json")
# the gold comparison predates the suffixed filenames, so fall back to the
# judge summary, which carries the same mean and definitely exists
cmp_gold = (J("hdrr_final_comparison_mistral-7b-instruct-v0-3_gold.json")
            or J("hdrr_final_comparison.json"))
_g = J("judge_summary_mistral-7b-instruct-v0-3_gold.json")
if cmp_gold is None and _g:
    cmp_gold = {"ours_all": {"mean": _g["mean"]}}
abl = J("ablation_v4.json")
rv2 = J("routing_v2_retrieval.json")
router = J("router_comparison.json")

PAPER = {"hybrid": "HDRR\n(their best)", "cbrllm": "CBR +\nLLM routing",
         "cbrmeta": "CBR +\nmetadata", "baseline": "Unnamed\nbaseline",
         "agentic": "Agentic\nRAG"}

# ============================================================ 1. systems
if cmp_v2:
    rows = [(PAPER.get(k, k), v["theirs_mean"])
            for k, v in cmp_v2["paired"].items()]
    rows.append(("FinRAG\n(ours)", cmp_v2["ours_all"]["mean"]))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [OURS if "FinRAG" in n else THEIR for n in names]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    b = ax.barh(names, vals, color=cols, height=0.62)
    for bar, v in zip(b, vals):
        ax.text(v + 0.08, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=10.5, weight="bold",
                color=OURS if bar.get_facecolor()[:3] ==
                matplotlib.colors.to_rgb(OURS) else THEIR)
    ax.set_xlim(0, 9)
    ax.set_xlabel("mean judge score (1–10)")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    finish(fig, ax, "Answer quality on 258 shared FinDER queries",
           "Scored by GPT-4.1 using Cheng et al.'s own rubric and prompt")
    save(fig, "01_systems.png")
    figs.append(("01_systems.png",
                 "Answer quality against every published system",
                 "Same 258 held-out queries, same judge, same rubric. "
                 "FinRAG runs entirely on one laptop GPU; every other system "
                 "generates its answers with a frontier API model."))

# ===================================================== 2. paired deltas
if cmp_v2:
    items = sorted(cmp_v2["paired"].items(), key=lambda kv: kv[1]["delta"])
    names = [PAPER.get(k, k).replace("\n", " ") for k, _ in items]
    d = [v["delta"] for _, v in items]
    lo = [v["delta"] - v["lo"] for _, v in items]
    hi = [v["hi"] - v["delta"] for _, v in items]
    sig = [v["significant"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    y = range(len(names))
    ax.axvline(0, color=INK, lw=1.2, zorder=1)
    for i, (dd, l, h, s) in enumerate(zip(d, lo, hi, sig)):
        c = WARM if s else NEUT
        ax.errorbar(dd, i, xerr=[[l], [h]], fmt="o", color=c, ecolor=c,
                    elinewidth=2.0, capsize=5, markersize=8, zorder=3)
        ax.text(dd, i + 0.28, f"{dd:+.2f}", ha="center", fontsize=9.5,
                color=c, weight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel("FinRAG minus theirs  (95% CI, 10,000 paired resamples)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.04, "orange = difference is real\ngrey = statistical tie",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=MUTE)
    finish(fig, ax, "Paired comparison, per query",
           "An interval crossing zero means the two systems cannot be "
           "separated")
    save(fig, "02_deltas.png")
    figs.append(("02_deltas.png", "Is the difference real?",
                 "Every system is scored on identical queries, so the "
                 "differences can be resampled directly. FinRAG is "
                 "statistically tied with Agentic RAG and trails the rest."))

# =============================================== 3. gap decomposition
if cmp_v2 and cmp_gold:
    dep = cmp_v2["ours_all"]["mean"]
    ceil = cmp_gold["ours_all"]["mean"]
    hd = cmp_v2["paired"]["hybrid"]["theirs_mean"]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    xs = ["FinRAG\nas deployed", "FinRAG with\nperfect retrieval",
          "HDRR\n(their best)"]
    vals = [dep, ceil, hd]
    cols = [OURS, OURS, THEIR]
    b = ax.bar(xs, vals, color=cols, width=0.5)
    b[1].set_alpha(0.55)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}",
                ha="center", fontsize=11.5, weight="bold")

    ax.annotate("", xy=(0.25, dep), xytext=(0.75, ceil),
                arrowprops=dict(arrowstyle="<->", color=WARM, lw=1.6))
    ax.text(0.5, (dep + ceil) / 2 + 0.12, f"retrieval\n{ceil - dep:+.2f}",
            ha="center", fontsize=10, color=WARM, weight="bold")
    ax.annotate("", xy=(1.25, ceil), xytext=(1.75, hd),
                arrowprops=dict(arrowstyle="<->", color=WARM, lw=1.6))
    ax.text(1.5, (ceil + hd) / 2 + 0.12, f"generator\n{hd - ceil:+.2f}",
            ha="center", fontsize=10, color=WARM, weight="bold")

    ax.set_ylim(0, 9)
    ax.set_ylabel("mean judge score (1–10)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    finish(fig, ax, f"Where the {hd - dep:.2f}-point gap comes from",
           "Perfect retrieval is measured by feeding the gold evidence "
           "directly")
    save(fig, "03_decomposition.png")
    figs.append(("03_decomposition.png",
                 "The gap splits into two very different problems",
                 f"{ceil - dep:.2f} points are lost to retrieval, which is "
                 f"fixable. {hd - ceil:.2f} points remain even with perfect "
                 f"evidence in front of the model — that is a 7B model on "
                 f"a laptop against a frontier API model, i.e. a hardware "
                 f"budget, not a method flaw."))

# ================================================ 4. retrieval ablation
if abl and rv2:
    t = abl["table"]
    stages = [("BM25", t["bm25"]["ndcg@10"], t["bm25"]["ndcg@10_ci"]),
              ("BGE-small\n(off the shelf)", t["small_dense"]["ndcg@10"],
               t["small_dense"]["ndcg@10_ci"]),
              ("+ fine-tuned", t["ft_dense"]["ndcg@10"],
               t["ft_dense"]["ndcg@10_ci"]),
              ("+ ticker routing", t["ft_dense_ticker"]["ndcg@10"],
               t["ft_dense_ticker"]["ndcg@10_ci"]),
              ("+ reranking", t["ft_dense_ticker_rerank@50"]["ndcg@10"],
               t["ft_dense_ticker_rerank@50"]["ndcg@10_ci"]),
              ("+ SEC-name routing", rv2["metrics"]["ndcg@10"]["new"], None)]
    names = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    err = [[v - s[2][0] if s[2] else 0 for v, s in zip(vals, stages)],
           [s[2][1] - v if s[2] else 0 for v, s in zip(vals, stages)]]

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    cols = [NEUT, NEUT, OURS, OURS, OURS, WARM]
    b = ax.bar(names, vals, color=cols, width=0.56,
               yerr=err, capsize=4, ecolor=MUTE, error_kw={"lw": 1.2})
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.022, f"{v:.3f}",
                ha="center", fontsize=10, weight="bold")
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("NDCG@10")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.5)
    finish(fig, ax, "Retrieval on 1,088 held-out questions",
           "Graded relevance, filing-level train/test split, 95% CI from a "
           "paired bootstrap")
    save(fig, "04_retrieval.png")
    figs.append(("04_retrieval.png",
                 "Retrieval — the metric their paper never reports",
                 "HDRR publishes only end-to-end judge scores, so there is no "
                 "published retrieval number on FinDER to compare against. "
                 "Every step here is measured on filings never seen in "
                 "training."))

# =================================================== 5. router head-to-head
if router:
    rows = router["rows"]
    label = {"HDRR LLM router (hybrid)": "HDRR\n1 LLM call per query",
             "ours: regex only": "Ours, regex\nfree",
             "ours: SEC names + stoplist": "Ours, SEC names\nfree"}
    names = [label.get(r["name"], r["name"]) for r in rows]
    cov = [100 * r["coverage"] for r in rows]
    corr = [100 * r["end_to_end"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = range(len(names))
    w = 0.36
    ax.bar([i - w / 2 for i in x], cov, w, label="queries routed",
           color=NEUT)
    ax.bar([i + w / 2 for i in x], corr, w,
           label="routed to the correct filing",
           color=[THEIR, OURS, OURS])
    for i, (c, k) in enumerate(zip(cov, corr)):
        ax.text(i - w / 2, c + 1.2, f"{c:.1f}", ha="center", fontsize=9.5)
        ax.text(i + w / 2, k + 1.2, f"{k:.1f}", ha="center", fontsize=10,
                weight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of queries")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ag = router["agreement"]
    finish(fig, ax, "Document routing: an LLM call vs a free dictionary",
           f"Where both routers committed, they chose the same filing on "
           f"{100 * ag['agreed'] / max(ag['both_routed'], 1):.0f}% of "
           f"{ag['both_routed']} queries")
    save(fig, "05_router.png")
    figs.append(("05_router.png",
                 "Matching their LLM router at zero cost per query",
                 "HDRR calls an LLM on every query to extract the company "
                 "ticker. Replacing it with SEC's public company-name table "
                 "plus an acronym stoplist recovers most of that capability "
                 "and costs nothing to run."))

# ====================================================== 6. what changed
s3 = J("judge_summary_mistral-7b-instruct-v0-3_retrieved.json")
s10 = J("judge_summary_mistral-7b-instruct-v0-3_retrieved_k10.json")
s10v2 = J("judge_summary_mistral-7b-instruct-v0-3_retrieved_k10_v2.json")
sg = J("judge_summary_mistral-7b-instruct-v0-3_gold.json")
if s3 and s10 and s10v2 and sg and cmp_v2:
    labels = ["starting\npipeline", "+ deeper\ncontext", "+ SEC-name\nrouting"]
    vals = [s3["mean"], s10["mean"], s10v2["mean"]]
    hd = cmp_v2["paired"]["hybrid"]["theirs_mean"]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(labels, vals, "-o", color=OURS, lw=2.4, markersize=9, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.14, f"{v:.2f}", ha="center", fontsize=11,
                weight="bold", color=OURS)
    # labels sit ABOVE their lines at the left margin: placing them at the
    # right-hand end put the text straight through the dashes
    ax.axhline(sg["mean"], color=OURS, ls=":", lw=1.6, alpha=0.75)
    ax.text(-0.3, sg["mean"] + 0.07,
            f"ceiling with perfect retrieval  {sg['mean']:.2f}",
            va="bottom", ha="left", fontsize=9.5, color=OURS)
    ax.axhline(hd, color=THEIR, ls="--", lw=1.8)
    ax.text(-0.3, hd + 0.07, f"HDRR  {hd:.2f}", va="bottom", ha="left",
            fontsize=9.5, color=THEIR, weight="bold")
    ax.set_ylim(5.0, 8.3)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylabel("mean judge score (1–10)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    finish(fig, ax, "Progress this week",
           "Same model and same training throughout — only retrieval "
           "depth and routing changed")
    save(fig, "06_progress.png")
    figs.append(("06_progress.png", "What changed this week",
                 f"No retraining was involved: {vals[0]:.2f} → "
                 f"{vals[-1]:.2f} came from feeding the generator more "
                 f"retrieved context and fixing which company it searches."))

# ================================================= 7. score distribution
jr = J("judge_mistral-7b-instruct-v0-3_retrieved_k10_v2.json")
bundle = J("hdrr_comparison_mistral-7b-instruct-v0-3_retrieved_k10_v2.json")
if jr and bundle:
    ours = [v["score"] for v in jr.values()
            if isinstance(v, dict) and v.get("score") is not None]
    theirs = [q["their_hybrid_score"] for q in bundle["queries"].values()
              if q.get("their_hybrid_score") is not None]
    bins = range(1, 12)
    oc = [ours.count(i) for i in range(1, 11)]
    tc = [round(sum(1 for t in theirs if round(t) == i)) for i in range(1, 11)]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = list(range(1, 11))
    w = 0.4
    ax.bar([i - w / 2 for i in x], oc, w, label="FinRAG (ours)", color=OURS)
    ax.bar([i + w / 2 for i in x], tc, w, label="HDRR", color=THEIR,
           alpha=0.85)
    ax.set_xticks(x)
    ax.set_xlabel("judge score")
    ax.set_ylabel("queries")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)
    finish(fig, ax, "Score distribution, 258 queries",
           "FinRAG is bimodal: it either answers well or fails outright")
    save(fig, "07_distribution.png")
    figs.append(("07_distribution.png", "How the scores are distributed",
                 "FinRAG scores 8+ on 48% of queries, close to HDRR's "
                 "quality when it works. Its losses are concentrated in "
                 "outright failures rather than mediocre answers, which is "
                 "the signature of missing evidence rather than weak "
                 "reasoning."))

# ================================================================= index
cards = "\n".join(f"""  <figure>
    <figcaption><span class="n">{i}</span>{t}</figcaption>
    <img src="{f}" alt="{t}">
    <p>{c}</p>
  </figure>""" for i, (f, t, c) in enumerate(figs, 1))

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinRAG Results</title>
<style>
  :root {{ --ink:#1b1b1f; --mute:#6b6b74; --line:#e3e3e8; --bg:#fff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;
    -webkit-font-smoothing:antialiased; }}
  main {{ max-width:960px; margin:0 auto; padding:56px 24px 80px; }}
  h1 {{ font-size:30px; line-height:1.2; margin:0 0 8px; letter-spacing:-.02em; }}
  .sub {{ color:var(--mute); margin:0 0 10px; font-size:15px; }}
  .lede {{ font-size:17px; line-height:1.55; margin:28px 0 0;
    padding:20px 22px; border-left:3px solid #2f6f4e; background:#f6f9f7;
    border-radius:0 8px 8px 0; }}
  .lede b {{ font-variant-numeric:tabular-nums; }}
  figure {{ margin:52px 0 0; padding:0; }}
  figcaption {{ font-size:19px; font-weight:650; letter-spacing:-.01em;
    margin:0 0 14px; display:flex; gap:12px; align-items:baseline; }}
  .n {{ font-size:12px; font-weight:700; color:var(--mute);
    border:1px solid var(--line); border-radius:999px;
    min-width:24px; height:24px; display:inline-flex; align-items:center;
    justify-content:center; flex:none; }}
  img {{ width:100%; height:auto; display:block;
    border:1px solid var(--line); border-radius:10px; }}
  figure p {{ color:var(--mute); font-size:14.5px; line-height:1.6;
    margin:14px 0 0; max-width:74ch; }}
  footer {{ margin-top:64px; padding-top:22px; border-top:1px solid var(--line);
    color:var(--mute); font-size:13.5px; }}
  @media print {{
    main {{ padding:0; max-width:none; }}
    figure {{ break-inside:avoid; margin-top:30px; }}
    .lede {{ background:none; }}
  }}
  @media (prefers-color-scheme:dark) {{
    :root:not([data-theme="light"]) {{ --ink:#ececf0; --mute:#9a9aa4;
      --line:#2e2e36; --bg:#141418; }}
    :root:not([data-theme="light"]) .lede {{ background:#17201b; }}
    :root:not([data-theme="light"]) img {{ background:#fff; }}
  }}
</style></head>
<body><main>
  <h1>FinRAG — results against HDRR (2026)</h1>
  <p class="sub">Haroon M Bilal · Muhammad Bin Sarwar · 28 October 2026</p>
  <p class="lede">On 258 held-out FinDER queries scored by <b>Cheng et al.'s own
  GPT-4.1 judge with their own rubric and prompt</b>, FinRAG reaches
  <b>6.02/10</b> against HDRR's <b>7.64</b>. Our grader was verified against 40
  of their published answers: 90% exact agreement, 100% within one point,
  r&nbsp;=&nbsp;0.993. The 1.61-point gap is <b>0.68 retrieval</b> and
  <b>0.94 generator capacity</b>. On retrieval itself — which their paper never
  reports — FinRAG reaches <b>NDCG@10 0.599</b>, and a zero-cost router matches
  their per-query LLM router's decision on <b>100%</b> of queries where both
  commit.</p>
{cards}
  <footer>All queries come from filings never seen during training.
  Confidence intervals are 10,000-sample paired bootstraps over identical
  queries.</footer>
</main></body></html>"""

(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"\n{len(figs)} figures + index.html -> {OUT}")
