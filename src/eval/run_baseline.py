"""
STAGE 2 — Baseline search + first real numbers.

Builds the search stack with OFF-THE-SHELF (generic, not fine-tuned) models —
the same recipe as the base paper — and scores it on your answer key.

It evaluates three rungs of the ladder:
    1) dense_only      : generic BGE embeddings only
    2) hybrid          : BGE + BM25 fused (RRF)
    3) hybrid_rerank   : hybrid + generic cross-encoder reranker  <- base-paper-style

Metrics (deterministic, no API): MRR@10, NDCG@10, Recall@5/10/20.

Run:  python src\\eval\\run_baseline.py
Output: results\\baseline.md  (+ printed table)

The GPU is used here (encoding + reranking). Everything else is read from the
three files Stage 1 produced.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from ranx import Qrels, Run, evaluate

# ── settings ───────────────────────────────────────────────────────────
PROC      = Path("data/finder/processed")
RESULTS   = Path("results"); RESULTS.mkdir(exist_ok=True)
EMBED     = "BAAI/bge-small-en-v1.5"       # generic embedder (baseline)
RERANKER  = "BAAI/bge-reranker-base"       # generic reranker (baseline)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DENSE_K, SPARSE_K, RRF_K, RERANK_K, STORE_K = 100, 100, 60, 50, 50
CUTOFFS   = [5, 10, 20]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
METRICS   = ["mrr@10", "ndcg@10"] + [f"recall@{k}" for k in CUTOFFS]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)
    return qrels


def rrf(dense, sparse, k=RRF_K):
    s = {}
    for hits in (dense, sparse):
        for r, (cid, _) in enumerate(hits):
            s[cid] = s.get(cid, 0.0) + 1.0 / (k + r + 1)
    return sorted(s.items(), key=lambda x: x[1], reverse=True)


def main():
    print(f"device: {DEVICE}")
    corpus = load_jsonl(PROC / "corpus.jsonl")
    queries = load_jsonl(PROC / "queries.jsonl")
    qrels_d = load_qrels(PROC / "qrels.tsv")
    # only score queries that have a gold chunk
    queries = [q for q in queries if str(q["_id"]) in qrels_d]
    cids = [c["_id"] for c in corpus]
    ctexts = [c["text"] for c in corpus]
    print(f"corpus={len(cids)}  scored_queries={len(queries)}")

    # ---- dense index (BGE) ----
    print("encoding corpus with BGE (GPU)...")
    embedder = SentenceTransformer(EMBED, device=DEVICE)
    cemb = embedder.encode(ctexts, batch_size=128, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)

    qtexts = [BGE_QUERY_PREFIX + q["text"] for q in queries]
    qemb = embedder.encode(qtexts, batch_size=128, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    # ---- sparse index (BM25) ----
    print("building BM25...")
    bm25 = BM25Okapi([t.lower().split() for t in ctexts])

    # ---- per-query retrieval ----
    print("retrieving...")
    cid2text = dict(zip(cids, ctexts))          # fast lookup (fixes the 3-hour bug)
    runs = {"dense_only": {}, "hybrid": {}, "hybrid_rerank": {}}
    reranker = CrossEncoder(RERANKER, device=DEVICE, max_length=512)
    t0 = time.time()
    for qi, q in enumerate(queries):
        qid = str(q["_id"])
        # dense
        sc, idx = index.search(qemb[qi:qi+1], DENSE_K)
        dense_hits = [(cids[i], float(s)) for i, s in zip(idx[0], sc[0]) if i != -1]
        # sparse
        bm = bm25.get_scores(q["text"].lower().split())
        top = np.argsort(bm)[::-1][:SPARSE_K]
        sparse_hits = [(cids[i], float(bm[i])) for i in top]
        # rung 1: dense only  (store top STORE_K -> valid recall@20)
        runs["dense_only"][qid] = {cid: s for cid, s in dense_hits[:STORE_K]}
        # rung 2: hybrid RRF
        fused = rrf(dense_hits, sparse_hits)
        runs["hybrid"][qid] = {cid: s for cid, s in fused[:STORE_K]}
        # rung 3: hybrid + rerank  (rerank top RERANK_K, keep all of them)
        cand_ids = [cid for cid, _ in fused[:RERANK_K]]
        pairs = [[q["text"], cid2text[cid]] for cid in cand_ids]
        rr = reranker.predict(pairs, batch_size=64)
        order = np.argsort(rr)[::-1]
        runs["hybrid_rerank"][qid] = {cand_ids[i]: float(rr[i]) for i in order}
        if qi % 200 == 0:
            print(f"  {qi}/{len(queries)}  ({time.time()-t0:.0f}s)", end="\r")
    print()

    # ---- score every rung ----
    qrels = Qrels(qrels_d)
    table = {}
    for name, run in runs.items():
        res = evaluate(qrels, Run(run), METRICS)
        table[name] = {m: float(res[m]) for m in METRICS}

    # ---- print + save ----
    header = "| System | " + " | ".join(METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    rows = [header, sep]
    for name, vals in table.items():
        rows.append("| " + name + " | " + " | ".join(f"{vals[m]:.4f}" for m in METRICS) + " |")
    out = "# Baseline (off-the-shelf models)\n\n" + "\n".join(rows) + "\n"
    print("\n" + out)
    (RESULTS / "baseline.md").write_text(out, encoding="utf-8")
    json.dump(table, open(RESULTS / "baseline.json", "w"), indent=2)
    print("saved -> results/baseline.md")


if __name__ == "__main__":
    main()