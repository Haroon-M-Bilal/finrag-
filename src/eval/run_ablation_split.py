"""
FULL ABLATION on held-out TEST filings (leakage-free).

Evaluates every retrieval configuration on the 1,104 test questions whose gold
chunks live in filings never seen during training. Retrieval still searches the
FULL corpus (realistic setting); only training was restricted.

Configurations (the two marked NEW were missing from the original paper and are
required to support the "reranking is redundant" claim):

    bm25                  sparse only                        NEW
    base_dense            off-the-shelf embedder
    base_hybrid           off-the-shelf + BM25 (RRF)
    base_hybrid_rerank    + off-the-shelf reranker
    ft_dense              fine-tuned embedder
    ft_dense_rerank       fine-tuned embedder + reranker      NEW  <- key cell
    ft_hybrid             fine-tuned + BM25 (RRF)
    ft_hybrid_rerank      + fine-tuned reranker

Reports MRR@10, NDCG@10, Recall@{5,10,20} with bootstrap 95% confidence
intervals, so "is this noise?" is answered directly.

Run:  python src\\eval\\run_ablation_split.py            (full, ~4-6 h)
      python src\\eval\\run_ablation_split.py 200        (quick check, 200 queries)
Output: results/ablation_split.md / .json
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
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
EMBED_BASE = "BAAI/bge-small-en-v1.5"
RERANK_BASE = "BAAI/bge-reranker-base"
EMBED_FT = str(CKPT / "bge-small-finder-split")
RERANK_FT = str(CKPT / "bge-reranker-finder-split")
PREFIX = "Represent this sentence for searching relevant passages: "
DENSE_K = SPARSE_K = 100
RRF_K, RERANK_K, STORE_K = 60, 50, 50
METRICS = ["mrr@10", "ndcg@10", "recall@5", "recall@10", "recall@20"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 300


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_qrels(p):
    q = {}
    f = open(p, encoding="utf-8"); next(f)
    for line in f:
        a, b, s = line.rstrip("\n").split("\t")
        q.setdefault(a, {})[b] = int(s)
    return q


def rrf(dense, sparse, k=RRF_K):
    s = {}
    for hits in (dense, sparse):
        for r, (cid, _) in enumerate(hits):
            s[cid] = s.get(cid, 0.0) + 1.0 / (k + r + 1)
    return sorted(s.items(), key=lambda x: x[1], reverse=True)


def bootstrap_ci(qrels, run, metric, n=N_BOOT, seed=0):
    """95% CI over queries, resampled with replacement."""
    rng = np.random.default_rng(seed)
    qids = [q for q in run if q in qrels]
    vals = []
    for _ in range(n):
        samp = rng.choice(qids, size=len(qids), replace=True)
        sub_q = {f"{i}": qrels[q] for i, q in enumerate(samp)}
        sub_r = {f"{i}": run[q] for i, q in enumerate(samp)}
        r = evaluate(Qrels(sub_q), Run(sub_r), [metric])
        vals.append(float(r[metric] if isinstance(r, dict) else r))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    queries = jl(PROC / "queries_test.jsonl")
    qrels = load_qrels(PROC / "qrels_test.tsv")
    queries = [q for q in queries if str(q["_id"]) in qrels]
    if limit:
        queries = queries[:limit]
        print(f"QUICK MODE: {len(queries)} queries")
    qrels = {str(q["_id"]): qrels[str(q["_id"])] for q in queries}
    print(f"TEST questions: {len(queries)}  corpus: {len(cids)}  device: {DEV}")

    print("building BM25 over full corpus...")
    bm25 = BM25Okapi([t.lower().split() for t in ctexts])

    def dense_index(path, cache):
        m = SentenceTransformer(path, device=DEV)
        if Path(cache).exists():
            cemb = np.load(cache)
        else:
            print(f"  encoding corpus for {path} ...")
            cemb = m.encode(ctexts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            np.save(cache, cemb)
        ix = faiss.IndexFlatIP(cemb.shape[1]); ix.add(cemb)
        qe = m.encode([PREFIX + q["text"] for q in queries], batch_size=128,
                      normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        del m
        return ix, qe

    print("preparing dense indexes...")
    ix_base, qe_base = dense_index(EMBED_BASE, str(PROC / "corpus_emb_base.npy"))
    ix_ft, qe_ft = dense_index(EMBED_FT, str(PROC / "corpus_emb_ft_split.npy"))

    rr_base = CrossEncoder(RERANK_BASE, device=DEV, max_length=512)
    rr_ft = CrossEncoder(RERANK_FT, device=DEV, max_length=512)

    runs = {k: {} for k in ["bm25", "base_dense", "base_hybrid", "base_hybrid_rerank",
                            "ft_dense", "ft_dense_rerank", "ft_hybrid", "ft_hybrid_rerank"]}
    t0 = time.time()
    for qi, q in enumerate(queries):
        qid = str(q["_id"]); qt = q["text"]

        bm = bm25.get_scores(qt.lower().split())
        top = np.argsort(bm)[::-1][:SPARSE_K]
        sparse_hits = [(cids[i], float(bm[i])) for i in top]
        runs["bm25"][qid] = {c: s for c, s in sparse_hits[:STORE_K]}

        for tag, ix, qe, rr in (("base", ix_base, qe_base, rr_base),
                                ("ft", ix_ft, qe_ft, rr_ft)):
            sc, idx = ix.search(qe[qi:qi+1], DENSE_K)
            dense_hits = [(cids[i], float(s)) for i, s in zip(idx[0], sc[0]) if i != -1]
            runs[f"{tag}_dense"][qid] = {c: s for c, s in dense_hits[:STORE_K]}

            fused = rrf(dense_hits, sparse_hits)
            runs[f"{tag}_hybrid"][qid] = {c: s for c, s in fused[:STORE_K]}

            # rerank the hybrid list
            cand = [c for c, _ in fused[:RERANK_K]]
            sco = rr.predict([[qt, cid2text[c]] for c in cand], batch_size=64)
            order = np.argsort(sco)[::-1]
            runs[f"{tag}_hybrid_rerank"][qid] = {cand[i]: float(sco[i]) for i in order}

            # NEW: rerank the DENSE-ONLY list (isolates reranking from fusion)
            if tag == "ft":
                cand_d = [c for c, _ in dense_hits[:RERANK_K]]
                sco_d = rr.predict([[qt, cid2text[c]] for c in cand_d], batch_size=64)
                od = np.argsort(sco_d)[::-1]
                runs["ft_dense_rerank"][qid] = {cand_d[i]: float(sco_d[i]) for i in od}

        if qi % 50 == 0:
            print(f"  {qi}/{len(queries)}  ({time.time()-t0:.0f}s)", end="\r")
    print()

    Q = Qrels(qrels)
    table = {}
    for name, run in runs.items():
        if not run:
            continue
        res = evaluate(Q, Run(run), METRICS)
        row = {m: float(res[m]) for m in METRICS}
        lo, hi = bootstrap_ci(qrels, run, "mrr@10")
        row["mrr@10_ci"] = [lo, hi]
        table[name] = row

    cols = METRICS
    lines = ["| System | " + " | ".join(cols) + " | MRR@10 95% CI |",
             "|" + "---|" * (len(cols) + 2)]
    for name, v in table.items():
        ci = v["mrr@10_ci"]
        lines.append("| " + name + " | " + " | ".join(f"{v[m]:.4f}" for m in cols) +
                     f" | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    out = (f"# Ablation on held-out TEST filings ({len(queries)} questions)\n\n"
           + "\n".join(lines) + "\n")
    print("\n" + out)
    if not limit:
        (RESULTS / "ablation_split.md").write_text(out, encoding="utf-8")
        json.dump(table, open(RESULTS / "ablation_split.json", "w"), indent=2)
        print("saved -> results/ablation_split.md")


if __name__ == "__main__":
    main()