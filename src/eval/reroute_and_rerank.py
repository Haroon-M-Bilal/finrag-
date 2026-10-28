"""
Rebuild the retrieval run file using the SEC-name router, and re-measure.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT

It reproduces exactly the pipeline that produced ft_dense_ticker_rerank@50 in
run_ablation_split.py - dense retrieval restricted to the routed filing, then
the fine-tuned cross-encoder over the top 50 - but using routing_v2.json
instead of the bare regex.

It does NOT re-run the whole ablation, because it does not need to:

  * The corpus embeddings are already cached (corpus_emb_ft_v4.npy), verified
    by fingerprint, so nothing is re-encoded.
  * Only queries whose ROUTING CHANGED can produce a different ranking. Every
    other query's candidate list is identical, so its existing run entries are
    copied across unchanged. That is typically ~150 of 1088 queries to
    recompute rather than all of them - minutes, not hours.
  * Queries that end up unrouted fall back to full-corpus dense retrieval,
    which is already on disk as ft_dense.trec. That is precisely what the
    original code did (`routed = dense["ft"][:STORE_K]`), so the fallback is
    reconstructed rather than recomputed.

A ROUTING CHANGE CAN GO EITHER WAY

The new rule drops ticker tokens in the acronym stoplist, so a query whose
only ticker token was "CF" or "NI" is now UNROUTED where it used to be routed.
That is intentional - those were the collisions - but it means the new rule is
not a strict superset and the comparison has to be measured, not assumed.

THE SCORES BELOW ARE A LOWER BOUND

qrels were built by prepare_finder_v4.py using the same colliding regex as an
override, so some gold labels point at the wrong filing (CF Industries for a
question about cash flow, and so on). A query whose label is wrong will now
route correctly, retrieve the right company's text, and score ZERO against a
bad label. The improvement reported here is therefore understated by however
much of that contamination remains.

Run:  python -u src/eval/reroute_and_rerank.py
      python -u src/eval/reroute_and_rerank.py --dry-run     (no GPU, counts only)
Output: results/runs/ft_dense_ticker_rerank_at_50_v2.trec
        results/routing_v2_retrieval.md / .json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROC = Path("data/finder/processed_v4")
CKPT = Path("checkpoints")
RESULTS = Path("results")
RUNDIR = RESULTS / "runs"

EMBED_FT = str(CKPT / "bge-small-finder-v4")
RERANK_FT = str(CKPT / "bge-reranker-finder-v4")
PREFIX = "Represent this sentence for searching relevant passages: "

OLD_RUN = RUNDIR / "ft_dense_ticker_rerank_at_50.trec"
DENSE_RUN = RUNDIR / "ft_dense.trec"
NEW_RUN = RUNDIR / "ft_dense_ticker_rerank_at_50_v2.trec"

STORE_K = 100
RERANK_K = 50
PASSAGE_TOKENS = 460          # identical to training and to the first ablation
N_BOOT = 10_000
SEED = 0

TICKRE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,5}\b")


# ------------------------------------------------------------------- io
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


def load_trec(p):
    """qid -> {cid: score}"""
    if not p.exists():
        raise SystemExit(f"missing {p} - run run_ablation_split.py first")
    run = defaultdict(dict)
    for line in open(p, encoding="utf-8"):
        qid, _, cid, _, score, _ = line.split()
        run[qid][cid] = float(score)
    return dict(run)


def write_trec(path, run, tag):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, d in run.items():
            for r, (cid, s) in enumerate(
                    sorted(d.items(), key=lambda kv: -kv[1]), 1):
                f.write(f"{qid} Q0 {cid} {r} {s:.6f} {tag}\n")


def corpus_fingerprint(cids) -> str:
    h = hashlib.sha1()
    for c in cids:
        h.update(c.encode())
    return h.hexdigest()


# -------------------------------------------------------------- metrics
def per_query_metrics(run, qrels, qid_order):
    """Identical to run_ablation_split.py so the numbers stay comparable."""
    out = {m: np.zeros(len(qid_order)) for m in
           ["mrr@10", "ndcg@10", "ndcg@20", "recall@10", "recall@100"]}
    for i, qid in enumerate(qid_order):
        gold = qrels.get(qid, {})
        if not gold:
            continue
        ranked = [c for c, _ in sorted(run.get(qid, {}).items(),
                                       key=lambda kv: -kv[1])]
        grades = [gold.get(c, 0) for c in ranked]
        for r, g in enumerate(grades[:10], 1):
            if g >= 1:
                out["mrr@10"][i] = 1.0 / r
                break
        ideal = sorted(gold.values(), reverse=True)
        for k, key in ((10, "ndcg@10"), (20, "ndcg@20")):
            dcg = sum(g / np.log2(r + 1)
                      for r, g in enumerate(grades[:k], 1))
            idcg = sum(g / np.log2(r + 1)
                       for r, g in enumerate(ideal[:k], 1))
            out[key][i] = dcg / idcg if idcg > 0 else 0.0
        n_gold = len(gold)
        for k in (10, 100):
            out[f"recall@{k}"][i] = sum(
                1 for c in ranked[:k] if c in gold) / n_gold
    return out


def paired(a_met, b_met, metric, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = a_met[metric], b_met[metric]
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = (a - b)[idx].mean(axis=1)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return float((a - b).mean()), lo, hi, bool(lo > 0 or hi < 0)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many queries would change, then exit")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    args = ap.parse_args()
    t_start = time.time()

    rv = PROC / "routing_v2.json"
    if not rv.exists():
        raise SystemExit(f"missing {rv} - run build_ticker_map.py first")
    new_routing = {k: v["ticker"]
                   for k, v in json.load(open(rv, encoding="utf-8"))["routing"].items()}

    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]
    ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    by_tick = defaultdict(list)
    for i, c in enumerate(cids):
        by_tick[c.split("::")[0]].append(i)
    all_ticks = set(by_tick)
    del corpus
    gc.collect()

    queries = jl(PROC / f"queries_{args.split}.jsonl")
    qrels = load_qrels(PROC / f"qrels_{args.split}.tsv")
    queries = [q for q in queries if str(q["_id"]) in qrels]
    qids = [str(q["_id"]) for q in queries]
    print(f"{args.split.upper()} queries: {len(queries)}   "
          f"corpus: {len(cids)} chunks")

    # ------------------------------------------- which queries actually move
    old_route, chg = {}, []
    for q in queries:
        qid = str(q["_id"])
        named = set(TICKRE.findall(q["text"])) & all_ticks
        o = named.pop() if len(named) == 1 else None
        n = new_routing.get(qid)
        old_route[qid] = o
        if o != n:
            chg.append((qid, o, n))

    gained = sum(1 for _, o, n in chg if o is None and n)
    lost = sum(1 for _, o, n in chg if o and n is None)
    moved = sum(1 for _, o, n in chg if o and n and o != n)
    print(f"\nrouting changes: {len(chg)}")
    print(f"  newly routed       : {gained}")
    print(f"  no longer routed   : {lost}   (acronym collisions removed)")
    print(f"  routed elsewhere   : {moved}")

    if args.dry_run:
        print("\ndry run - nothing computed.")
        return
    if not chg:
        raise SystemExit("no routing changed; nothing to rebuild")

    old_run = load_trec(OLD_RUN)
    dense_run = load_trec(DENSE_RUN)

    # --------------------------------------------------- recompute the movers
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    npy = PROC / "corpus_emb_ft_v4.npy"
    meta = PROC / "corpus_emb_ft_v4.meta.json"
    if not (npy.exists() and meta.exists()):
        raise SystemExit(f"missing {npy} / {meta}")
    m = json.load(open(meta, encoding="utf-8"))
    if (m.get("fingerprint") != corpus_fingerprint(cids)
            or m.get("n_chunks") != len(cids)):
        raise SystemExit(
            f"\nCACHE MISMATCH for {npy}: it was built for a different "
            f"corpus.\nDelete it and re-run run_ablation_split.py.\n")
    print(f"\nloading cached corpus embeddings ({len(cids)} chunks, verified)")
    cemb = np.load(npy)

    rr_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    _clip: dict[str, str] = {}

    def clip(text: str) -> str:
        hit = _clip.get(text)
        if hit is not None:
            return hit
        ids = rr_tok.encode(text, add_special_tokens=False)
        out = (rr_tok.decode(ids[:PASSAGE_TOKENS], skip_special_tokens=True)
               if len(ids) > PASSAGE_TOKENS else text)
        _clip[text] = out
        return out

    qtext = {str(q["_id"]): q["text"] for q in queries}
    movers = [qid for qid, _, _ in chg]

    print(f"encoding {len(movers)} changed queries...")
    emb_model = SentenceTransformer(EMBED_FT, device=dev)
    qemb = emb_model.encode([PREFIX + qtext[q] for q in movers],
                            batch_size=128, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
    del emb_model
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    print("reranking...")
    rr = CrossEncoder(RERANK_FT, device=dev, max_length=512)
    new_run = dict(old_run)
    t0 = time.time()

    for i, (qid, _, n) in enumerate(chg):
        if n and n in by_tick:
            fidx = by_tick[n]
            sims = cemb[fidx] @ qemb[i]
            order = np.argsort(-sims)[:STORE_K]
            cand = [(cids[fidx[int(o)]], float(sims[int(o)])) for o in order]
        else:
            # unrouted: full-corpus dense, exactly as the original fallback
            cand = sorted(dense_run.get(qid, {}).items(),
                          key=lambda kv: -kv[1])[:STORE_K]
        if not cand:
            continue
        ids = [c for c, _ in cand[:RERANK_K]]
        sco = rr.predict([[qtext[qid], clip(cid2text[c])] for c in ids],
                         batch_size=64, show_progress_bar=False)
        new_run[qid] = {c: float(s) for c, s in zip(ids, sco)}
        if i % 20 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(chg)}  elapsed {el:.0f}s  "
                  f"eta {el / max(i, 1) * (len(chg) - i):.0f}s", end="\r")
    print(f"\n  recomputed in {time.time() - t0:.0f}s")

    del rr
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    write_trec(NEW_RUN, new_run, "ft_dense_ticker_rerank@50_v2")
    print(f"wrote -> {NEW_RUN}")

    # --------------------------------------------------------------- score
    a = per_query_metrics(new_run, qrels, qids)
    b = per_query_metrics(old_run, qrels, qids)
    METRICS = ["ndcg@10", "ndcg@20", "mrr@10", "recall@10", "recall@100"]

    L = [f"# Routing v2: SEC company names + acronym stoplist", "",
         f"Held-out {args.split} questions ({len(qids)}). Same pipeline as "
         f"`ft_dense_ticker_rerank@50` - dense retrieval inside the routed "
         f"filing, then the fine-tuned cross-encoder over the top "
         f"{RERANK_K} - with routing from SEC's public company-name mapping "
         f"instead of a bare ticker regex. {len(chg)} queries changed routing "
         f"({gained} newly routed, {lost} de-routed as acronym collisions, "
         f"{moved} re-targeted); the rest are byte-identical.", "",
         "**These deltas are a lower bound.** `prepare_finder_v4.py` resolved "
         "questions to filings using the same colliding regex, so some gold "
         "labels name the wrong company. A query whose label is wrong now "
         "routes correctly and scores zero against it.", "",
         "| metric | regex routing | SEC-name routing | delta | 95% CI | sig |",
         "|---|---|---|---|---|---|"]

    summary = {}
    for met in METRICS:
        d, lo, hi, sig = paired(a, b, met)
        summary[met] = {"old": float(b[met].mean()),
                        "new": float(a[met].mean()),
                        "delta": d, "lo": lo, "hi": hi, "significant": sig}
        L.append(f"| {met} | {b[met].mean():.4f} | {a[met].mean():.4f} | "
                 f"{d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                 f"{'**yes**' if sig else 'no'} |")

    # the movers in isolation - where the whole effect lives
    midx = [qids.index(q) for q, _, _ in chg if q in qids]
    if midx:
        L += ["", f"## The {len(midx)} queries whose routing changed", "",
              "Everything else is unchanged by construction, so the aggregate "
              "delta above is this effect diluted across the full test set.",
              "", "| metric | before | after | delta |", "|---|---|---|---|"]
        for met in METRICS:
            ov, nv = b[met][midx].mean(), a[met][midx].mean()
            L.append(f"| {met} | {ov:.4f} | {nv:.4f} | {nv - ov:+.4f} |")
        summary["movers_only"] = {
            met: {"old": float(b[met][midx].mean()),
                  "new": float(a[met][midx].mean())} for met in METRICS}

    out = "\n".join(L) + "\n"
    print("\n" + out)
    (RESULTS / "routing_v2_retrieval.md").write_text(out, encoding="utf-8")
    json.dump({"split": args.split, "n_queries": len(qids),
               "n_changed": len(chg), "newly_routed": gained,
               "de_routed": lost, "re_targeted": moved,
               "metrics": summary, "n_boot": N_BOOT, "seed": SEED,
               "new_run_file": str(NEW_RUN)},
              open(RESULTS / "routing_v2_retrieval.json", "w"), indent=2)
    print(f"saved -> results/routing_v2_retrieval.md / .json")
    print(f"total wall clock: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()