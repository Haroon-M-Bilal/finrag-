"""
STAGE 3d — Evaluate FINE-TUNED models and compare to the stock baseline.

Two modes:
  QUICK SANITY (recommended first):
      python src\\eval\\run_finetuned.py 300
      -> scores STOCK vs FINE-TUNED on the same first 300 questions. Fast.
         Use this only to confirm the numbers went up. NOT for the paper.

  FULL (the real result):
      python src\\eval\\run_finetuned.py
      -> scores fine-tuned on all 5,625 questions, merges with results/baseline.json,
         writes results/comparison.md  (the table for your paper).

Caches fine-tuned corpus embeddings to corpus_emb_ft.npy so a later full run is faster.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from ranx import Qrels, Run, evaluate

PROC      = Path("data/finder/processed")
RESULTS   = Path("results"); RESULTS.mkdir(exist_ok=True)
CKPT      = Path("checkpoints")
EMBED_STOCK = "BAAI/bge-small-en-v1.5"
RERANK_STOCK = "BAAI/bge-reranker-base"
EMBED_FT  = str(CKPT / "bge-small-finder")
RERANK_FT = str(CKPT / "bge-reranker-finder")
PREFIX    = "Represent this sentence for searching relevant passages: "
DENSE_K, SPARSE_K, RRF_K, RERANK_K, STORE_K = 100, 100, 60, 50, 50
CUTOFFS   = [5, 10, 20]
METRICS   = ["mrr@10", "ndcg@10"] + [f"recall@{k}" for k in CUTOFFS]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_qrels(p):
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, s = line.rstrip("\n").split("\t")
            q.setdefault(qid, {})[cid] = int(s)
    return q


def rrf(dense, sparse, k=RRF_K):
    s = {}
    for hits in (dense, sparse):
        for r, (cid, _) in enumerate(hits):
            s[cid] = s.get(cid, 0.0) + 1.0 / (k + r + 1)
    return sorted(s.items(), key=lambda x: x[1], reverse=True)


def eval_system(tag, embed_path, reranker_path, queries, cids, ctexts, cid2text,
                bm25, qrels, cache_emb=None):
    """Returns {rung_name: {metric: val}} for one (embedder, reranker) pair."""
    print(f"\n=== {tag} ===")
    embedder = SentenceTransformer(embed_path, device=DEVICE)

    if cache_emb and Path(cache_emb).exists():
        print("  loading cached corpus embeddings...")
        cemb = np.load(cache_emb)
    else:
        print("  encoding corpus (GPU)...")
        cemb = embedder.encode(ctexts, batch_size=128, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        if cache_emb:
            np.save(cache_emb, cemb)
    index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)

    qemb = embedder.encode([PREFIX + q["text"] for q in queries],
                           batch_size=128, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    reranker = CrossEncoder(reranker_path, device=DEVICE, max_length=512)
    runs = {f"{tag}_dense": {}, f"{tag}_hybrid": {}, f"{tag}_hybrid_rerank": {}}
    t0 = time.time()
    for qi, q in enumerate(queries):
        qid = str(q["_id"])
        sc, idx = index.search(qemb[qi:qi+1], DENSE_K)
        dense_hits = [(cids[i], float(s)) for i, s in zip(idx[0], sc[0]) if i != -1]
        bm = bm25.get_scores(q["text"].lower().split())
        top = np.argsort(bm)[::-1][:SPARSE_K]
        sparse_hits = [(cids[i], float(bm[i])) for i in top]
        runs[f"{tag}_dense"][qid] = {cid: s for cid, s in dense_hits[:STORE_K]}
        fused = rrf(dense_hits, sparse_hits)
        runs[f"{tag}_hybrid"][qid] = {cid: s for cid, s in fused[:STORE_K]}
        cand = [cid for cid, _ in fused[:RERANK_K]]
        rr = reranker.predict([[q["text"], cid2text[c]] for c in cand], batch_size=64)
        order = np.argsort(rr)[::-1]
        runs[f"{tag}_hybrid_rerank"][qid] = {cand[i]: float(rr[i]) for i in order}
        if qi % 200 == 0:
            print(f"  {qi}/{len(queries)} ({time.time()-t0:.0f}s)", end="\r")
    print()
    qrels_obj = Qrels(qrels)
    return {name: {m: float(evaluate(qrels_obj, Run(run), METRICS)[m]) for m in METRICS}
            for name, run in runs.items()}


def to_table(table: dict) -> str:
    header = "| System | " + " | ".join(METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    rows = [header, sep]
    for name, vals in table.items():
        rows.append("| " + name + " | " + " | ".join(f"{vals[m]:.4f}" for m in METRICS) + " |")
    return "\n".join(rows)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    corpus = load_jsonl(PROC / "corpus.jsonl")
    queries = load_jsonl(PROC / "queries.jsonl")
    qrels_all = load_qrels(PROC / "qrels.tsv")
    queries = [q for q in queries if str(q["_id"]) in qrels_all]
    if limit:
        queries = queries[:limit]
        print(f"QUICK SANITY MODE: first {len(queries)} questions")
    qrels = {str(q["_id"]): qrels_all[str(q["_id"])] for q in queries}
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    print(f"corpus={len(cids)}  scored_queries={len(queries)}  device={DEVICE}")

    print("building BM25...")
    bm25 = BM25Okapi([t.lower().split() for t in ctexts])

    # fine-tuned system (cache its corpus embeddings for reuse)
    ft = eval_system("ft", EMBED_FT, RERANK_FT, queries, cids, ctexts, cid2text,
                     bm25, qrels, cache_emb=str(PROC / "corpus_emb_ft.npy"))

    if limit:
        # also score STOCK on the same subset for a fair quick comparison
        stock = eval_system("stock", EMBED_STOCK, RERANK_STOCK, queries, cids, ctexts,
                            cid2text, bm25, qrels, cache_emb=str(PROC / "corpus_emb_base.npy"))
        combined = {**stock, **ft}
        print("\n# QUICK SANITY (same " + str(len(queries)) + " questions)\n")
        print(to_table(combined))
        print("\n(quick check only — run without a number for the full paper table)")
    else:
        baseline = json.load(open(RESULTS / "baseline.json"))   # stock, full set
        combined = {**baseline, **ft}
        out = "# Baseline vs Fine-tuned (full eval, 5,625 questions)\n\n" + to_table(combined) + "\n"
        (RESULTS / "comparison.md").write_text(out, encoding="utf-8")
        json.dump(combined, open(RESULTS / "comparison.json", "w"), indent=2)
        print("\n" + out)
        print("saved -> results/comparison.md")


if __name__ == "__main__":
    main()