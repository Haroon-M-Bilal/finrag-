"""
FULL ABLATION on held-out TEST filings (leakage-free).

Evaluates retrieval configurations on the test questions whose gold chunks live
in filings never seen during training. Retrieval still searches the FULL corpus
(the realistic setting); only training was restricted.

DESIGN NOTES

  1. PER-QUERY RUN FILES ARE DUMPED TO DISK (results/runs/*.trec, TREC format).
     Every later metric question - paired bootstrap, NDCG@10 vs @20, linear vs
     exponential gain, threshold sensitivity, per-category breakdowns - becomes
     offline re-scoring of these files instead of another multi-hour GPU run.

  2. METRICS COMPUTED DIRECTLY, NOT VIA ranx AGGREGATES, so the gain convention
     is explicit and stateable in the paper and per-query scores are available
     for the paired test. ranx is used at the end only to cross-check.

  3. PAIRED BOOTSTRAP. One set of resampled query indices is shared across
     systems, so per-system CIs and pairwise deltas come from the same
     resamples. Independent per-system bootstraps ignore that every system is
     scored on the same queries and badly overstate uncertainty on a difference.

  4. RERANK DEPTH SWEEP (50 / 100 / 200), because a fixed RERANK_K measures pool
     depth as much as reranking.

  5. TICKER-ROUTED RETRIEVAL. Only ~11% of the fine-tuned retriever's top-100
     come from the question's own filing - 10-K filings are structurally
     near-identical, so most retrieved text is the wrong company. Most FinDER
     queries name the ticker in plain text, so restricting the pool to that
     filing is a string match: document routing with no LLM and no API.

  6. BGE-BASE BASELINE, so "fine-tuning helps" is not confounded with "the base
     model was undersized".

  7. PASSAGES CLIPPED IN THE RERANKER'S OWN TOKENIZER, identically to training.
     Chunks are 448 BGE WordPiece tokens but measure mean 475 / p90 525 under
     bge-reranker-base's XLM-RoBERTa, so ~15% would otherwise lose their tail.

CHANGES AFTER THE FIRST FULL RUN DIED SILENTLY AT ~28 MIN

  a. MEMORY. The small/base corpus embedding arrays are released once their
     FAISS indexes exist (FAISS holds its own copy); only the fine-tuned array
     is still needed, for ticker routing. Frees ~850 MB.
  b. ONE CROSS-ENCODER RESIDENT AT A TIME. The base reranker is now run in a
     separate pass after the fine-tuned one is released, rather than both being
     held for the whole loop.
  c. INCREMENTAL CHECKPOINTS. Partial runs are written to disk every
     CHECKPOINT_EVERY queries, so a crash costs minutes rather than the whole
     job.
  d. RESUME. Pass --resume to reload the last checkpoint and continue.

Run:  python -u src\\eval\\run_ablation_split.py                (full)
      python -u src\\eval\\run_ablation_split.py 200            (timing check)
      python -u src\\eval\\run_ablation_split.py --resume       (continue)
Output: results/ablation_v4.md / .json, results/runs/*.trec
"""
from __future__ import annotations

import gc
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer

PROC = Path("data/finder/processed_v4")
CKPT = Path("checkpoints")
RESULTS = Path("results")
RUNDIR = RESULTS / "runs"
RESULTS.mkdir(exist_ok=True)
RUNDIR.mkdir(parents=True, exist_ok=True)

EMBED_SMALL = "BAAI/bge-small-en-v1.5"
EMBED_BASE = "BAAI/bge-base-en-v1.5"
RERANK_BASE = "BAAI/bge-reranker-base"
EMBED_FT = str(CKPT / "bge-small-finder-v4")
RERANK_FT = str(CKPT / "bge-reranker-finder-v4")
PREFIX = "Represent this sentence for searching relevant passages: "

DENSE_K = 200
SPARSE_K = 200
RRF_K = 60
RERANK_DEPTHS = [50, 100, 200]
STORE_K = 100
N_BOOT = 1000
SEED = 0
PASSAGE_TOKENS = 460
CHECKPOINT_EVERY = 100

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TICKRE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,5}\b")
STATE = RESULTS / "ablation_state.pkl"

_rr_tok = AutoTokenizer.from_pretrained(RERANK_BASE)
_clip_cache: dict[str, str] = {}


def clip(text: str) -> str:
    """Truncate in the RERANKER's tokenizer, matching training exactly."""
    hit = _clip_cache.get(text)
    if hit is not None:
        return hit
    ids = _rr_tok.encode(text, add_special_tokens=False)
    out = (_rr_tok.decode(ids[:PASSAGE_TOKENS], skip_special_tokens=True)
           if len(ids) > PASSAGE_TOKENS else text)
    _clip_cache[text] = out
    return out


# ----------------------------------------------------------------- io helpers
def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_qrels(p):
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, s = line.rstrip("\n").split("\t")
            q.setdefault(a, {})[b] = int(s)
    return q


def corpus_fingerprint(cids) -> str:
    h = hashlib.sha1()
    for c in cids:
        h.update(c.encode())
    return h.hexdigest()


def load_or_build(model_path, model, cids, ctexts, tag):
    npy = PROC / f"corpus_emb_{tag}.npy"
    meta = PROC / f"corpus_emb_{tag}.meta.json"
    fp = corpus_fingerprint(cids)
    if npy.exists() and meta.exists():
        m = json.load(open(meta, encoding="utf-8"))
        if (m.get("fingerprint") != fp or m.get("n_chunks") != len(cids)
                or m.get("model") != model_path):
            raise SystemExit(
                f"\nCACHE MISMATCH for {npy}\n"
                f"  cached : {m.get('n_chunks')} chunks, {m.get('model')}\n"
                f"  current: {len(cids)} chunks, {model_path}\n"
                f"Delete {npy} and {meta} and re-run.\n")
        print(f"  cache hit: {tag} ({len(cids)} chunks, verified)")
        return np.load(npy)
    if npy.exists():
        raise SystemExit(f"\n{npy} has no manifest and cannot be verified. "
                         f"Delete it and re-run.\n")
    print(f"  encoding corpus with {model_path} ...")
    cemb = model.encode(ctexts, batch_size=128, normalize_embeddings=True,
                        show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)
    np.save(npy, cemb)
    json.dump({"model": model_path, "n_chunks": len(cids),
               "dim": int(cemb.shape[1]), "fingerprint": fp},
              open(meta, "w"), indent=2)
    return cemb


def write_trec(path, run, tag):
    with open(path, "w", encoding="utf-8") as f:
        for qid, d in run.items():
            for r, (cid, s) in enumerate(
                    sorted(d.items(), key=lambda kv: -kv[1]), 1):
                f.write(f"{qid} Q0 {cid} {r} {s:.6f} {tag}\n")


# --------------------------------------------------------------------- metrics
def per_query_metrics(run, qrels, qid_order):
    """
    MRR@10   reciprocal rank of the first chunk with grade >= 1, else 0
    NDCG@k   DCG@k / IDCG@k; gains linear (g) and exponential (2^g - 1).
             IDCG uses the query's own graded gold sorted descending.
    Recall@k |top-k retrieved that are gold| / |gold|
    """
    out = {m: np.zeros(len(qid_order)) for m in
           ["mrr@10", "ndcg@10", "ndcg@20", "ndcg_exp@10",
            "recall@5", "recall@10", "recall@20", "recall@100"]}

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
        for k, key, exp in ((10, "ndcg@10", False), (20, "ndcg@20", False),
                            (10, "ndcg_exp@10", True)):
            gain = (lambda g: (2 ** g - 1)) if exp else (lambda g: g)
            dcg = sum(gain(g) / np.log2(r + 1)
                      for r, g in enumerate(grades[:k], 1))
            idcg = sum(gain(g) / np.log2(r + 1)
                       for r, g in enumerate(ideal[:k], 1))
            out[key][i] = dcg / idcg if idcg > 0 else 0.0

        n_gold = len(gold)
        for k in (5, 10, 20, 100):
            hit = sum(1 for c in ranked[:k] if c in gold)
            out[f"recall@{k}"][i] = hit / n_gold

    return out


def paired_bootstrap(per_sys, qid_order, metric, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_q = len(qid_order)
    idx = rng.integers(0, n_q, size=(n, n_q))
    boot, ci = {}, {}
    for name, mets in per_sys.items():
        b = mets[metric][idx].mean(axis=1)
        boot[name] = b
        ci[name] = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    return ci, boot


def delta_ci(boot, a, b):
    d = boot[a] - boot[b]
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return float(d.mean()), lo, hi, (lo > 0 or hi < 0)


# ------------------------------------------------------------------------ main
def main():
    args = sys.argv[1:]
    resume = "--resume" in args
    nums = [a for a in args if a.isdigit()]
    limit = int(nums[0]) if nums else None
    t_start = time.time()

    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]
    ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    by_tick = {}
    for i, c in enumerate(cids):
        by_tick.setdefault(c.split("::")[0], []).append(i)
    all_ticks = set(by_tick)
    del corpus
    gc.collect()

    queries = jl(PROC / "queries_test.jsonl")
    qrels_all = load_qrels(PROC / "qrels_test.tsv")
    queries = [q for q in queries if str(q["_id"]) in qrels_all]
    if limit:
        queries = queries[:limit]
        print(f"QUICK MODE: {len(queries)} queries")
    qids = [str(q["_id"]) for q in queries]
    qrels = {q: qrels_all[q] for q in qids}
    print(f"TEST questions: {len(queries)}  corpus: {len(cids)}  device: {DEV}")

    names = ["bm25", "small_dense", "base_dense", "ft_dense",
             "ft_hybrid", "ft_hybrid_rerank@50",
             "ft_dense_rerank@50", "ft_dense_rerank@100", "ft_dense_rerank@200",
             "ft_dense_rerank_base@50", "ft_dense_ticker",
             "ft_dense_ticker_rerank@50"]

    runs = {n: {} for n in names}
    start_at = 0
    if resume and STATE.exists():
        with open(STATE, "rb") as f:
            st = pickle.load(f)
        if st.get("n_queries") == len(queries):
            runs = st["runs"]
            start_at = st["done"]
            print(f"RESUMED from checkpoint at query {start_at}")
        else:
            print("checkpoint query count differs - starting fresh")

    print("\nbuilding BM25 over full corpus...")
    t0 = time.time()
    bm25 = BM25Okapi([t.lower().split() for t in ctexts])
    print(f"  BM25 built in {time.time() - t0:.0f}s")

    print("\npreparing dense indexes...")
    indexes, qembs = {}, {}
    emb_ft = None
    for tag, path in (("small", EMBED_SMALL), ("base", EMBED_BASE),
                      ("ft", EMBED_FT)):
        m = SentenceTransformer(path, device=DEV)
        cache_tag = {"small": "base", "base": "bgebase", "ft": "ft_v4"}[tag]
        e = load_or_build(path, m, cids, ctexts, cache_tag)
        qembs[tag] = m.encode([PREFIX + q["text"] for q in queries],
                              batch_size=128, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
        ix = faiss.IndexFlatIP(e.shape[1])
        ix.add(e)
        indexes[tag] = ix
        if tag == "ft":
            emb_ft = e          # still needed for ticker routing
        else:
            del e               # FAISS holds its own copy
        del m
        gc.collect()
        torch.cuda.empty_cache()
    print("  released non-ft corpus arrays")

    q_tick, n_routed = {}, 0
    for q in queries:
        named = set(TICKRE.findall(q["text"])) & all_ticks
        if len(named) == 1:
            q_tick[str(q["_id"])] = named.pop()
            n_routed += 1
    print(f"\nticker named in query: {n_routed}/{len(queries)} "
          f"({100 * n_routed / len(queries):.1f}%)")

    # ---------------------------------------------- pass 1: FT reranker only
    rr_ft = CrossEncoder(RERANK_FT, device=DEV, max_length=512)
    print("\nretrieving (pass 1: dense, sparse, hybrid, ft reranker)...")
    t0 = time.time()
    dense_ft_cache: dict[str, list] = {}

    for qi in range(start_at, len(queries)):
        q = queries[qi]
        qid = qids[qi]
        qt = q["text"]

        bm = bm25.get_scores(qt.lower().split())
        top = np.argpartition(-bm, SPARSE_K)[:SPARSE_K]
        top = top[np.argsort(-bm[top])]
        sparse_hits = [(cids[i], float(bm[i])) for i in top]
        runs["bm25"][qid] = dict(sparse_hits[:STORE_K])

        dense = {}
        for tag in ("small", "base", "ft"):
            sc, idx = indexes[tag].search(qembs[tag][qi:qi + 1], DENSE_K)
            dense[tag] = [(cids[i], float(s))
                          for i, s in zip(idx[0], sc[0]) if i != -1]
        runs["small_dense"][qid] = dict(dense["small"][:STORE_K])
        runs["base_dense"][qid] = dict(dense["base"][:STORE_K])
        runs["ft_dense"][qid] = dict(dense["ft"][:STORE_K])
        dense_ft_cache[qid] = dense["ft"][:50]

        fused = {}
        for hits in (dense["ft"], sparse_hits):
            for r, (cid, _) in enumerate(hits):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r + 1)
        fused_l = sorted(fused.items(), key=lambda x: -x[1])
        runs["ft_hybrid"][qid] = dict(fused_l[:STORE_K])

        def rerank(cand_ids, model, out_name):
            if not cand_ids:
                return
            sco = model.predict([[qt, clip(cid2text[c])] for c in cand_ids],
                                batch_size=64, show_progress_bar=False)
            runs[out_name][qid] = {c: float(s) for c, s in zip(cand_ids, sco)}

        for d in RERANK_DEPTHS:
            rerank([c for c, _ in dense["ft"][:d]], rr_ft,
                   f"ft_dense_rerank@{d}")
        rerank([c for c, _ in fused_l[:50]], rr_ft, "ft_hybrid_rerank@50")

        t = q_tick.get(qid)
        if t and t in by_tick:
            fidx = by_tick[t]
            sims = emb_ft[fidx] @ qembs["ft"][qi]
            order = np.argsort(-sims)[:STORE_K]
            routed = [(cids[fidx[int(o)]], float(sims[int(o)])) for o in order]
        else:
            routed = dense["ft"][:STORE_K]
        runs["ft_dense_ticker"][qid] = dict(routed)
        rerank([c for c, _ in routed[:50]], rr_ft, "ft_dense_ticker_rerank@50")

        if qi % 25 == 0:
            el = time.time() - t0
            done = qi - start_at + 1
            eta = el / max(done, 1) * (len(queries) - qi)
            print(f"  {qi}/{len(queries)}  elapsed {el:.0f}s  eta {eta:.0f}s",
                  end="\r")
        if qi % CHECKPOINT_EVERY == 0 and qi > start_at:
            with open(STATE, "wb") as f:
                pickle.dump({"runs": runs, "done": qi,
                             "n_queries": len(queries)}, f)
    print(f"\n  pass 1 done in {time.time() - t0:.0f}s")

    del rr_ft
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------- pass 2: base reranker control
    rr_base = CrossEncoder(RERANK_BASE, device=DEV, max_length=512)
    print("\npass 2: off-the-shelf reranker control at depth 50...")
    t0 = time.time()
    for qi, q in enumerate(queries):
        qid = qids[qi]
        cand = [c for c, _ in dense_ft_cache.get(qid, [])]
        if not cand:
            continue
        sco = rr_base.predict([[q["text"], clip(cid2text[c])] for c in cand],
                              batch_size=64, show_progress_bar=False)
        runs["ft_dense_rerank_base@50"][qid] = {
            c: float(s) for c, s in zip(cand, sco)}
        if qi % 50 == 0:
            print(f"  {qi}/{len(queries)}", end="\r")
    print(f"\n  pass 2 done in {time.time() - t0:.0f}s")

    del rr_base
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ run files
    for name, run in runs.items():
        if run:
            write_trec(RUNDIR / f"{name.replace('@', '_at_')}.trec", run, name)
    print(f"wrote {len([r for r in runs.values() if r])} run files -> {RUNDIR}")

    # -------------------------------------------------------------- scoring
    per_sys = {n: per_query_metrics(r, qrels, qids)
               for n, r in runs.items() if r}

    METRICS = ["ndcg@10", "ndcg@20", "mrr@10", "recall@10", "recall@100"]
    ci, boot = paired_bootstrap(per_sys, qids, "ndcg@10")

    table = {}
    for n, mets in per_sys.items():
        table[n] = {m: float(mets[m].mean()) for m in mets}
        table[n]["ndcg@10_ci"] = list(ci[n])

    try:
        from ranx import Qrels, Run, evaluate
        probe = "ft_dense"
        rx = float(evaluate(Qrels(qrels), Run(runs[probe]), "ndcg@10"))
        print(f"\nranx cross-check on {probe}: ranx {rx:.4f} vs ours "
              f"{table[probe]['ndcg@10']:.4f}  "
              f"(diff {abs(rx - table[probe]['ndcg@10']):.4f})")
    except Exception as e:  # noqa: BLE001
        print(f"\nranx cross-check skipped: {e}")

    pairs = [("ft_dense", "small_dense"), ("ft_dense", "base_dense"),
             ("ft_dense", "bm25"), ("ft_hybrid", "ft_dense"),
             ("ft_dense_rerank@50", "ft_dense"),
             ("ft_dense_rerank@200", "ft_dense_rerank@50"),
             ("ft_dense_rerank@50", "ft_dense_rerank_base@50"),
             ("ft_dense_ticker", "ft_dense"),
             ("ft_dense_ticker_rerank@50", "ft_dense_ticker")]
    deltas = []
    for a, b in pairs:
        if a in boot and b in boot:
            m, lo, hi, sig = delta_ci(boot, a, b)
            deltas.append({"a": a, "b": b, "delta": m, "lo": lo, "hi": hi,
                           "significant": bool(sig)})

    L = [f"# Ablation on held-out TEST filings ({len(queries)} questions)", "",
         f"Corpus {len(cids)} chunks, {PROC}. NDCG uses LINEAR gain "
         f"(g / log2(r+1)); ndcg_exp uses (2^g - 1). Graded qrels: grade 2 = "
         f"primary evidence chunk, grade 1 = supporting. Ticker named in "
         f"{n_routed}/{len(queries)} queries "
         f"({100 * n_routed / len(queries):.1f}%).", "",
         "| System | " + " | ".join(METRICS) + " | NDCG@10 95% CI |",
         "|" + "---|" * (len(METRICS) + 2)]
    for n in names:
        if n not in table:
            continue
        v = table[n]
        c = v["ndcg@10_ci"]
        L.append("| " + n + " | " + " | ".join(f"{v[m]:.4f}" for m in METRICS)
                 + f" | [{c[0]:.4f}, {c[1]:.4f}] |")

    L += ["", "## Paired bootstrap on NDCG@10 (same resamples across systems)",
          "", "| A | B | A-B | 95% CI | significant |", "|---|---|---|---|---|"]
    for d in deltas:
        L.append(f"| {d['a']} | {d['b']} | {d['delta']:+.4f} | "
                 f"[{d['lo']:+.4f}, {d['hi']:+.4f}] | "
                 f"{'yes' if d['significant'] else 'no'} |")

    out = "\n".join(L) + "\n"
    print("\n" + out)
    print(f"total wall clock: {time.time() - t_start:.0f}s")

    if not limit:
        (RESULTS / "ablation_v4.md").write_text(out, encoding="utf-8")
        json.dump({"table": table, "deltas": deltas,
                   "n_queries": len(queries), "n_chunks": len(cids),
                   "dense_k": DENSE_K, "sparse_k": SPARSE_K,
                   "rerank_depths": RERANK_DEPTHS, "n_boot": N_BOOT,
                   "seed": SEED, "passage_tokens": PASSAGE_TOKENS,
                   "ticker_named_frac": n_routed / len(queries)},
                  open(RESULTS / "ablation_v4.json", "w"), indent=2)
        print("saved -> results/ablation_v4.md, results/ablation_v4.json")
        if STATE.exists():
            STATE.unlink()


if __name__ == "__main__":
    main()