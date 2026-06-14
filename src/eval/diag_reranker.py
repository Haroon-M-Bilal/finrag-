"""
DIAGNOSTIC — why did the fine-tuned reranker hurt?
On the first N questions, with FINE-TUNED embeddings fixed, compare:
    ft_norerank   : fine-tuned hybrid, no reranking
    ft+stock_rr   : + STOCK reranker
    ft+ft_rr      : + FINE-TUNED reranker
Reuses cached ft embeddings -> fast (no re-encoding).

Run:  python src\\eval\\diag_reranker.py 300
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import faiss, torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from ranx import Qrels, Run, evaluate

PROC = Path("data/finder/processed"); CKPT = Path("checkpoints")
PREFIX = "Represent this sentence for searching relevant passages: "
DENSE_K = SPARSE_K = 100; RRF_K = 60; RERANK_K = STORE_K = 50
METRICS = ["mrr@10", "ndcg@10", "recall@5", "recall@10", "recall@20"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_qrels(p):
    q = {}
    f = open(p, encoding="utf-8"); next(f)
    for line in f:
        a, b, c = line.rstrip("\n").split("\t"); q.setdefault(a, {})[b] = int(c)
    return q


def rrf(d, s, k=RRF_K):
    o = {}
    for hits in (d, s):
        for r, (cid, _) in enumerate(hits):
            o[cid] = o.get(cid, 0.0) + 1.0 / (k + r + 1)
    return sorted(o.items(), key=lambda x: x[1], reverse=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    corpus = jl(PROC / "corpus.jsonl"); queries = jl(PROC / "queries.jsonl")
    qr = load_qrels(PROC / "qrels.tsv")
    queries = [q for q in queries if str(q["_id"]) in qr][:n]
    qrels = {str(q["_id"]): qr[str(q["_id"])] for q in queries}
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    print(f"questions={len(queries)} device={DEV}")

    emb = SentenceTransformer(str(CKPT / "bge-small-finder"), device=DEV)
    cemb = np.load(PROC / "corpus_emb_ft.npy")           # cached -> fast
    index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)
    qemb = emb.encode([PREFIX + q["text"] for q in queries], batch_size=128,
                      normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    print("building BM25..."); bm25 = BM25Okapi([t.lower().split() for t in ctexts])
    stock_rr = CrossEncoder("BAAI/bge-reranker-base", device=DEV, max_length=512)
    ft_rr = CrossEncoder(str(CKPT / "bge-reranker-finder"), device=DEV, max_length=512)

    runs = {"ft_norerank": {}, "ft+stock_rr": {}, "ft+ft_rr": {}}
    t0 = time.time()
    for qi, q in enumerate(queries):
        qid = str(q["_id"])
        sc, idx = index.search(qemb[qi:qi+1], DENSE_K)
        dh = [(cids[i], float(s)) for i, s in zip(idx[0], sc[0]) if i != -1]
        bm = bm25.get_scores(q["text"].lower().split())
        sh = [(cids[i], float(bm[i])) for i in np.argsort(bm)[::-1][:SPARSE_K]]
        fused = rrf(dh, sh)
        runs["ft_norerank"][qid] = {cid: s for cid, s in fused[:STORE_K]}
        cand = [cid for cid, _ in fused[:RERANK_K]]
        pairs = [[q["text"], cid2text[c]] for c in cand]
        for tag, rr in (("ft+stock_rr", stock_rr), ("ft+ft_rr", ft_rr)):
            sco = rr.predict(pairs, batch_size=64)
            order = np.argsort(sco)[::-1]
            runs[tag][qid] = {cand[i]: float(sco[i]) for i in order}
        if qi % 100 == 0:
            print(f"  {qi}/{len(queries)} ({time.time()-t0:.0f}s)", end="\r")
    print()
    Q = Qrels(qrels)
    print("\n| System | " + " | ".join(METRICS) + " |")
    print("|" + "---|" * (len(METRICS) + 1))
    for name, run in runs.items():
        r = evaluate(Q, Run(run), METRICS)
        print("| " + name + " | " + " | ".join(f"{float(r[m]):.4f}" for m in METRICS) + " |")


if __name__ == "__main__":
    main()