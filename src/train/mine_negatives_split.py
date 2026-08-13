"""
Mine hard negatives using TRAIN-SPLIT questions only (leakage-free).

Uses the BASE embedder to mine negatives for the first embedder fine-tune.
Only questions in queries_train.jsonl are used, so no test-filing question ever
enters training. Negatives may come from any filing (that is realistic - the
retriever searches the whole corpus), but the query-positive pairs are train-only.

Run:  python src\\train\\mine_negatives_split.py
Output: data/finder/processed/train_triples_split.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss, torch
from sentence_transformers import SentenceTransformer

PROC = Path("data/finder/processed")
EMBED = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
TOP_N, N_NEG = 50, 8
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
    print(f"TRAIN questions: {len(queries)}  corpus: {len(cids)}  device: {DEV}")

    model = SentenceTransformer(EMBED, device=DEV)
    emb_path = PROC / "corpus_emb_base.npy"
    if emb_path.exists():
        print("loading cached base corpus embeddings...")
        cemb = np.load(emb_path)
    else:
        print("encoding corpus with base BGE...")
        cemb = model.encode(ctexts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(emb_path, cemb)
    index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)

    qemb = model.encode([PREFIX + q["text"] for q in queries], batch_size=128,
                        normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)

    print("mining hard negatives (train split only)...")
    n = 0
    with open(PROC / "train_triples_split.jsonl", "w", encoding="utf-8") as out:
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
    print(f"DONE. wrote {n} triples -> train_triples_split.jsonl")


if __name__ == "__main__":
    main()