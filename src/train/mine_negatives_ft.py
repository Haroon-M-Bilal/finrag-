"""
Re-mine hard negatives using the FINE-TUNED embedder (retriever-consistent).
The reranker will run behind the fine-tuned retriever, so its negatives must come
from THAT retriever's top results -- the genuinely hard, all-relevant chunks.

Reuses cached corpus_emb_ft.npy -> no re-encoding -> fast (~3 min).

Run:  python src\\train\\mine_negatives_ft.py
Output: data/finder/processed/train_triples_rr.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss, torch
from sentence_transformers import SentenceTransformer

PROC = Path("data/finder/processed"); CKPT = Path("checkpoints")
PREFIX = "Represent this sentence for searching relevant passages: "
TOP_N, N_NEG = 50, 6
DEV = "cuda" if torch.cuda.is_available() else "cpu"

corpus = [json.loads(l) for l in open(PROC / "corpus.jsonl", encoding="utf-8")]
queries = [json.loads(l) for l in open(PROC / "queries.jsonl", encoding="utf-8")]
qr = {}
f = open(PROC / "qrels.tsv", encoding="utf-8"); next(f)
for line in f:
    a, b, c = line.rstrip("\n").split("\t"); qr.setdefault(a, set()).add(b)
queries = [q for q in queries if str(q["_id"]) in qr]
cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
cid2text = dict(zip(cids, ctexts))
print(f"scored_queries={len(queries)} device={DEV}")

emb = SentenceTransformer(str(CKPT / "bge-small-finder"), device=DEV)
cemb = np.load(PROC / "corpus_emb_ft.npy")
index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)
qemb = emb.encode([PREFIX + q["text"] for q in queries], batch_size=128,
                  normalize_embeddings=True, show_progress_bar=True,
                  convert_to_numpy=True).astype(np.float32)

print("mining retriever-consistent negatives...")
n = 0
with open(PROC / "train_triples_rr.jsonl", "w", encoding="utf-8") as out:
    sc, idx = index.search(qemb, TOP_N)
    for qi, q in enumerate(queries):
        qid = str(q["_id"]); gold = qr[qid]
        pos = cid2text[sorted(gold)[0]]
        negs = [cid2text[cids[j]] for j in idx[qi] if cids[j] not in gold][:N_NEG]
        if not negs:
            continue
        out.write(json.dumps({"query": q["text"], "positive": pos, "negatives": negs}) + "\n")
        n += 1
print(f"DONE. wrote {n} -> train_triples_rr.jsonl")