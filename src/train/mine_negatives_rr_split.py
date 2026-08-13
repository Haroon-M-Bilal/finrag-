"""
Mine reranker negatives from the SPLIT-TRAINED embedder (train questions only).

The reranker runs behind the fine-tuned retriever at inference, so its negatives
must come from that retriever's own top results - otherwise it is trained on easy
negatives and collapses at test time (this happened in the first version of the
pipeline).

Encodes the corpus with the v4 fine-tuned embedder and caches it to
corpus_emb_ft_v4.npy, which the ablation evaluation reuses.

CHANGES FROM THE PREVIOUS VERSION:

  1. POSITIVES USE GRADES. The old version took sorted(gold)[0] - the
     alphabetically first chunk id, so "AAPL::ch12" sorted before "AAPL::ch3"
     and the positive was arbitrary. Now one row per GRADE-2 (primary) chunk.

  2. EXCLUSION BAND, DELIBERATELY LOOSER THAN THE EMBEDDER'S. Gold chunks and
     their adjacent neighbours (which share CHUNK_OVERLAP tokens by
     construction) are always excluded. Beyond that the threshold is 0.50
     rather than the embedder's 0.35: rejecting plausible-but-wrong passages is
     precisely what a cross-encoder is for, so filtering near-misses too
     aggressively removes the examples it most needs to see.

  3. NO FORCED IN-FILING / CROSS-FILING RATIO. The embedder searches the whole
     corpus, so its negatives were balanced deliberately. The reranker only ever
     sees what the retriever hands it, so its training distribution should be
     that pool as it actually is. The breakdown is recorded as a statistic
     rather than imposed.

  4. TOP_N RAISED TO 100 so training covers the deeper candidate pools used in
     the reranker-depth sweep (50 / 100 / 200) at evaluation time.

  5. READS processed_v4 and checkpoints/bge-small-finder-v4, with a
     manifest-guarded embedding cache that hard-fails on corpus mismatch.

Run:  python -u src\\train\\mine_negatives_rr_split.py
Output: data/finder/processed_v4/train_triples_rr_split.jsonl
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROC = Path("data/finder/processed_v4")
CKPT = Path("checkpoints")
EMBED_FT = str(CKPT / "bge-small-finder-v4")
PREFIX = "Represent this sentence for searching relevant passages: "

TOP_N = 100             # candidate pool per query, from the FT retriever
N_NEG = 6               # negatives kept per query
EXCLUDE_OVERLAP = 0.50  # looser than the embedder's band - see note 2 above
ADJACENT = 1

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOKRE = re.compile(r"[a-z0-9]{4,}")


def content(t: str) -> Counter:
    return Counter(TOKRE.findall(t.lower()))


def load_qrels(p):
    """qid -> {cid: grade}"""
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, g = line.rstrip("\n").split("\t")
            q.setdefault(a, {})[b] = int(g)
    return q


def corpus_fingerprint(cids) -> str:
    h = hashlib.sha1()
    for c in cids:
        h.update(c.encode())
    return h.hexdigest()


def load_or_build_embeddings(model, cids, ctexts):
    npy = PROC / "corpus_emb_ft_v4.npy"
    meta = PROC / "corpus_emb_ft_v4.meta.json"
    fp = corpus_fingerprint(cids)

    if npy.exists() and meta.exists():
        m = json.load(open(meta, encoding="utf-8"))
        ok = (m.get("fingerprint") == fp
              and m.get("n_chunks") == len(cids)
              and m.get("model") == EMBED_FT)
        if not ok:
            raise SystemExit(
                f"\nCACHE MISMATCH: {npy} was built for a different corpus or "
                f"model.\n  cached : {m.get('n_chunks')} chunks, {m.get('model')}"
                f"\n  current: {len(cids)} chunks, {EMBED_FT}\n"
                f"Delete {npy} and {meta} and re-run.\n")
        print(f"loading cached FT corpus embeddings ({len(cids)} chunks, verified)...")
        return np.load(npy)

    if npy.exists():
        raise SystemExit(
            f"\n{npy} exists but has no manifest, so it cannot be verified. "
            f"Delete it and re-run.\n")

    print("encoding corpus with the v4 fine-tuned embedder (~9 min)...")
    cemb = model.encode(ctexts, batch_size=128, normalize_embeddings=True,
                        show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)
    np.save(npy, cemb)
    json.dump({"model": EMBED_FT, "n_chunks": len(cids), "dim": int(cemb.shape[1]),
               "fingerprint": fp}, open(meta, "w"), indent=2)
    return cemb


def main():
    corpus = [json.loads(l) for l in open(PROC / "corpus.jsonl", encoding="utf-8")]
    cids = [c["_id"] for c in corpus]
    ctexts = [c["text"] for c in corpus]
    cid2i = {c: i for i, c in enumerate(cids)}

    queries = [json.loads(l) for l in open(PROC / "queries_train.jsonl",
                                           encoding="utf-8")]
    qrels = load_qrels(PROC / "qrels_train.tsv")
    print(f"TRAIN questions: {len(queries)}  corpus: {len(cids)}  device: {DEV}")

    emb = SentenceTransformer(EMBED_FT, device=DEV)
    cemb = load_or_build_embeddings(emb, cids, ctexts)

    index = faiss.IndexFlatIP(cemb.shape[1])
    index.add(cemb)

    qemb = emb.encode([PREFIX + q["text"] for q in queries], batch_size=128,
                      normalize_embeddings=True, show_progress_bar=True,
                      convert_to_numpy=True).astype(np.float32)

    print("mining retriever-consistent negatives...")
    _, idxs = index.search(qemb, TOP_N)

    n_rows = 0
    n_band = 0
    n_short = 0
    in_filing_total = 0
    neg_total = 0
    gold_in_pool = 0

    with open(PROC / "train_triples_rr_split.jsonl", "w", encoding="utf-8") as out:
        for qi, q in enumerate(queries):
            qid = str(q["_id"])
            gold = qrels.get(qid, {})
            if not gold:
                continue

            tick = next(iter(gold)).split("::")[0]
            banned = {cid2i[c] for c in gold if c in cid2i}
            for c in gold:
                t, ch = c.split("::ch")
                for d in range(1, ADJACENT + 1):
                    for nb in (f"{t}::ch{int(ch) - d}", f"{t}::ch{int(ch) + d}"):
                        if nb in cid2i:
                            banned.add(cid2i[nb])

            gold_content = Counter()
            for c in gold:
                if c in cid2i:
                    gold_content += content(ctexts[cid2i[c]])

            pool = [int(j) for j in idxs[qi]]
            if any(cids[j] in gold for j in pool):
                gold_in_pool += 1

            negs = []
            for j in pool:
                if j in banned:
                    continue
                cc = content(ctexts[j])
                tot = sum(cc.values())
                if tot == 0:
                    continue
                if sum((cc & gold_content).values()) / tot >= EXCLUDE_OVERLAP:
                    n_band += 1
                    continue
                negs.append(j)
                if len(negs) >= N_NEG:
                    break

            if not negs:
                continue
            if len(negs) < N_NEG:
                n_short += 1

            n_in = sum(1 for j in negs if cids[j].split("::")[0] == tick)
            in_filing_total += n_in
            neg_total += len(negs)

            primaries = [c for c, g in gold.items() if g == 2] or list(gold)
            for c in primaries:
                if c not in cid2i:
                    continue
                out.write(json.dumps({
                    "qid": qid,
                    "query": q["text"],
                    "pos_id": c,
                    "positive": ctexts[cid2i[c]],
                    "negatives": [ctexts[j] for j in negs],
                    "n_in_filing": n_in,
                    "n_cross_filing": len(negs) - n_in,
                }) + "\n")
                n_rows += 1

    json.dump({
        "source": str(PROC),
        "retriever": EMBED_FT,
        "top_n": TOP_N,
        "n_neg": N_NEG,
        "exclude_overlap": EXCLUDE_OVERLAP,
        "adjacent_excluded": ADJACENT,
        "forced_filing_ratio": False,
        "n_train_questions": len(queries),
        "n_rows": n_rows,
        "candidates_dropped_by_band": n_band,
        "queries_short_of_negs": n_short,
        "in_filing_negative_frac": round(in_filing_total / max(neg_total, 1), 3),
        "queries_with_gold_in_top_n": gold_in_pool,
        "gold_recall_at_top_n": round(gold_in_pool / max(len(queries), 1), 3),
    }, open(PROC / "mining_stats_rr.json", "w"), indent=2)

    print(f"\nDONE. wrote {n_rows} rows -> train_triples_rr_split.jsonl")
    print(f"  candidates dropped by band : {n_band}")
    print(f"  queries short of negatives : {n_short}")
    print(f"  in-filing negative fraction: "
          f"{in_filing_total / max(neg_total, 1):.3f}")
    print(f"  gold recall @ top-{TOP_N}      : "
          f"{gold_in_pool / max(len(queries), 1):.3f}  "
          f"({gold_in_pool}/{len(queries)} train queries)")


if __name__ == "__main__":
    main()