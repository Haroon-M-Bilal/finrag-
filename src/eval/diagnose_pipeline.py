"""
PIPELINE DIAGNOSTIC - find the real bottleneck before rebuilding anything.

Answers four questions, cheaply, using data already on disk:

  1. EMBEDDING SPACE GEOMETRY
     Mean pairwise cosine similarity of random chunk pairs.
     ~0.8+  -> space is collapsed (anisotropy): everything looks similar to
               everything, so ranking signal is weak. Chunking alone won't fix it.
     ~0.2-0.4 -> space is healthy; the problem is what's inside the chunks.

  2. GOLD-CHUNK SEPARATION
     For test queries, compare the query-gold similarity against the
     query-top1-retrieved similarity. If gold scores nearly as high as the
     retrieved top-1 but still ranks low, the space cannot separate near-ties.

  3. WHERE THE GOLD CHUNK ACTUALLY RANKS
     Distribution of the gold chunk's rank. If most golds sit at rank 1000+,
     the retriever is failing outright. If many sit at 20-100, a deeper
     candidate pool or reranking would recover them.

  4. TABLE / NUMERIC CONTENT CHECK
     How much of the corpus and how many gold chunks are number-dense.
     If gold chunks are far more numeric than the corpus average, blind
     300-word chunking is likely shredding tables.

Run:  python src\\eval\\diagnose_pipeline.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import faiss, torch
from sentence_transformers import SentenceTransformer

PROC = Path("data/finder/processed"); CKPT = Path("checkpoints")
EMBED_FT = str(CKPT / "bge-small-finder-split")
PREFIX = "Represent this sentence for searching relevant passages: "
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLE = 2000
N_QUERIES = 300


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_qrels(p):
    q = {}
    f = open(p, encoding="utf-8"); next(f)
    for line in f:
        a, b, _ = line.rstrip("\n").split("\t")
        q.setdefault(a, set()).add(b)
    return q


def numeric_density(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    nums = sum(1 for t in toks if re.search(r"\d", t))
    return nums / len(toks)


def main():
    rng = np.random.default_rng(0)
    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    cid2ix = {c: i for i, c in enumerate(cids)}
    queries = jl(PROC / "queries_test.jsonl")
    qrels = load_qrels(PROC / "qrels_test.tsv")
    queries = [q for q in queries if str(q["_id"]) in qrels][:N_QUERIES]
    print(f"corpus: {len(cids)}   test queries sampled: {len(queries)}   device: {DEV}")

    cemb = np.load(PROC / "corpus_emb_ft_split.npy")
    print(f"embeddings: {cemb.shape}  (dim={cemb.shape[1]})")

    # ---------- 1. geometry ----------
    print("\n" + "=" * 55)
    print("1. EMBEDDING SPACE GEOMETRY")
    idx = rng.choice(len(cemb), size=min(N_SAMPLE, len(cemb)), replace=False)
    S = cemb[idx]
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
    sims = S @ S.T
    iu = np.triu_indices_from(sims, k=1)
    vals = sims[iu]
    mean_sim = float(vals.mean())
    print(f"  mean pairwise cosine : {mean_sim:.4f}")
    print(f"  std                  : {float(vals.std()):.4f}")
    print(f"  5th / 95th pct       : {float(np.percentile(vals,5)):.4f} / {float(np.percentile(vals,95)):.4f}")
    if mean_sim > 0.75:
        print("  => COLLAPSED SPACE. Anisotropy is the bottleneck.")
        print("     Fix: whitening, larger training batch, or a higher-dim model.")
    elif mean_sim > 0.5:
        print("  => MODERATELY COMPRESSED. Geometry contributes but isn't the whole story.")
    else:
        print("  => HEALTHY SPREAD. Geometry is not the bottleneck; look at chunk content.")

    # ---------- 2 & 3. gold separation and rank ----------
    print("\n" + "=" * 55)
    print("2/3. GOLD CHUNK SEPARATION AND RANK")
    m = SentenceTransformer(EMBED_FT, device=DEV)
    qemb = m.encode([PREFIX + q["text"] for q in queries], batch_size=128,
                    normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)
    K = 1000
    scores, idxs = index.search(qemb, K)

    ranks, gold_sims, top1_sims = [], [], []
    for i, q in enumerate(queries):
        gold = qrels[str(q["_id"])]
        gix = [cid2ix[g] for g in gold if g in cid2ix]
        if not gix:
            continue
        gold_sims.append(float(max(qemb[i] @ cemb[g] for g in gix)))
        top1_sims.append(float(scores[i][0]))
        pos = [r for r, j in enumerate(idxs[i]) if cids[j] in gold]
        ranks.append(pos[0] + 1 if pos else K + 1)

    ranks = np.array(ranks)
    print(f"  mean query-gold  similarity : {np.mean(gold_sims):.4f}")
    print(f"  mean query-top1  similarity : {np.mean(top1_sims):.4f}")
    print(f"  gap (top1 - gold)           : {np.mean(top1_sims) - np.mean(gold_sims):.4f}")
    print(f"\n  gold chunk rank distribution:")
    for lo, hi in [(1, 1), (2, 10), (11, 20), (21, 100), (101, 1000)]:
        n = int(((ranks >= lo) & (ranks <= hi)).sum())
        print(f"    rank {lo:>4}-{hi:<4}: {n:>4}  ({100*n/len(ranks):5.1f}%)")
    n_miss = int((ranks > K).sum())
    print(f"    beyond {K:<7}: {n_miss:>4}  ({100*n_miss/len(ranks):5.1f}%)")
    recov = int(((ranks > 20) & (ranks <= 100)).sum())
    print(f"\n  recoverable by deeper pool (rank 21-100): {recov} ({100*recov/len(ranks):.1f}%)")
    if n_miss / len(ranks) > 0.4:
        print("  => Most golds are NOT retrievable at all. Content/representation problem,")
        print("     not a ranking problem. Reranking cannot help these.")

    # ---------- 4. numeric density ----------
    print("\n" + "=" * 55)
    print("4. NUMERIC / TABLE CONTENT")
    samp = rng.choice(len(ctexts), size=min(3000, len(ctexts)), replace=False)
    corpus_nd = float(np.mean([numeric_density(ctexts[i]) for i in samp]))
    gold_ids = {g for q in queries for g in qrels[str(q["_id"])]}
    gold_nd = float(np.mean([numeric_density(ctexts[cid2ix[g]])
                             for g in gold_ids if g in cid2ix]))
    print(f"  mean numeric density, corpus      : {corpus_nd:.4f}")
    print(f"  mean numeric density, gold chunks : {gold_nd:.4f}")
    if gold_nd > corpus_nd * 1.3:
        print("  => Gold chunks are markedly more numeric than average.")
        print("     Blind word-count chunking is likely splitting tables. Fix chunking.")
    else:
        print("  => Gold chunks are not unusually numeric; table shredding is less likely")
        print("     to be the dominant issue.")

    print("\n" + "=" * 55)
    print("Use the three verdicts above to choose ONE fix, then re-measure.")


if __name__ == "__main__":
    main()