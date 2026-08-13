"""
Mine reranker negatives from the SPLIT-TRAINED embedder (train questions only).

The reranker runs behind the fine-tuned retriever at inference, so its negatives
must come from that retriever's own top results - otherwise it is trained on easy
negatives and collapses at test time (this happened in the first version of the
pipeline).

Encodes the corpus with the split embedder and caches it to corpus_emb_ft_split.npy,
which the ablation evaluation reuses.

Run:  python src\\train\\mine_negatives_rr_split.py
Output: data/finder/processed/train_triples_rr_split.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss, torch
from sentence_transformers import SentenceTransformer

PROC = Path("data/finder/processed"); CKPT = Path("checkpoints")
EMBED_FT = str(CKPT / "bge-small-finder-split")
PREFIX = "Represent this sentence for searching relevant passages: "
TOP_N, N_NEG = 50, 6
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_qrels(p):
    q = {}
    f = open(p, encoding="utf-8"); next(f)
    for line in f:
        a, b, _ = line.rstrip("\n").split("\t")
        q.setdefault(a, set()).add(b)
    return q


def main():
    corpus = [json.loads(l) for l in open(PROC / "corpus.jsonl", encoding="utf-8")]
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    queries = [json.loads(l) for l in open(PROC / "queries_train.jsonl", encoding="utf-8")]
    qrels = load_qrels(PROC / "qrels_train.tsv")
    print(f"TRAIN questions: {len(queries)}  device: {DEV}")

    emb = SentenceTransformer(EMBED_FT, device=DEV)
    cache = PROC / "corpus_emb_ft_split.npy"
    if cache.exists():
        print("loading cached split-embedder corpus embeddings...")
        cemb = np.load(cache)
    else:
        print("encoding corpus with split embedder (one-time, ~10 min)...")
        cemb = emb.encode(ctexts, batch_size=128, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(cache, cemb)
    index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)

    qemb = emb.encode([PREFIX + q["text"] for q in queries], batch_size=128,
                      normalize_embeddings=True, show_progress_bar=True,
                      convert_to_numpy=True).astype(np.float32)

    print("mining retriever-consistent negatives...")
    n = 0
    with open(PROC / "train_triples_rr_split.jsonl", "w", encoding="utf-8") as out:
        _, idxs = index.search(qemb, TOP_N)
        for qi, q in enumerate(queries):
            qid = str(q["_id"]); gold = qrels.get(qid, set())
            if not gold:
                continue
            pos = cid2text[sorted(gold)[0]]
            negs = [cid2text[cids[j]] for j in idxs[qi] if cids[j] not in gold][:N_NEG]
            if not negs:
                continue
            out.write(json.dumps({"query": q["text"], "positive": pos,
                                  "negatives": negs}) + "\n")
            n += 1
    print(f"DONE. wrote {n} -> train_triples_rr_split.jsonl")
    print(f"cached embeddings -> {cache}")


if __name__ == "__main__":
    main()