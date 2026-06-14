"""
STAGE 3a — Build fine-tuning data (hard-negative mining).

For every question we create training examples of the form:
    (query, correct_chunk, [wrong-but-tempting chunks])

The "wrong-but-tempting" chunks (HARD NEGATIVES) are mined with the BASE model:
we retrieve its top hits and keep the high-ranked ones that are NOT the gold
chunk. Training the model to separate the gold chunk from these is what makes
domain fine-tuning actually move the numbers.

Also saves the base-model corpus embeddings (corpus_emb_base.npy) so later steps
don't have to re-encode from scratch.

Run:  python src\\train\\build_training_data.py
Output:
    data/finder/processed/train_triples.jsonl   # {"query","positive","negatives":[...]}
    data/finder/processed/corpus_emb_base.npy    # cached base embeddings
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

PROC      = Path("data/finder/processed")
EMBED     = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_N     = 50          # how deep to look when mining
N_NEG     = 8           # hard negatives per query
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
            q.setdefault(qid, set()).add(cid)
    return q


def main():
    print(f"device: {DEVICE}")
    corpus = load_jsonl(PROC / "corpus.jsonl")
    queries = load_jsonl(PROC / "queries.jsonl")
    qrels = load_qrels(PROC / "qrels.tsv")
    cids = [c["_id"] for c in corpus]
    ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    queries = [q for q in queries if str(q["_id"]) in qrels]
    print(f"corpus={len(cids)}  scored_queries={len(queries)}")

    model = SentenceTransformer(EMBED, device=DEVICE)

    # encode corpus once, cache to disk
    emb_path = PROC / "corpus_emb_base.npy"
    if emb_path.exists():
        print("loading cached corpus embeddings...")
        cemb = np.load(emb_path)
    else:
        print("encoding corpus with base BGE (GPU)...")
        cemb = model.encode(ctexts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(emb_path, cemb)
    index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)

    # encode queries
    print("encoding queries...")
    qemb = model.encode([BGE_QUERY_PREFIX + q["text"] for q in queries],
                        batch_size=128, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    print("mining hard negatives...")
    n_written = 0
    with open(PROC / "train_triples.jsonl", "w", encoding="utf-8") as out:
        scores, idxs = index.search(qemb, TOP_N)
        for qi, q in enumerate(queries):
            qid = str(q["_id"])
            gold = qrels[qid]
            positive = cid2text[sorted(gold)[0]]      # one gold chunk as the positive
            negs = []
            for j in idxs[qi]:
                cid = cids[j]
                if cid in gold:
                    continue                          # never use a gold chunk as negative
                negs.append(cid2text[cid])
                if len(negs) >= N_NEG:
                    break
            if not negs:
                continue
            out.write(json.dumps({"query": q["text"], "positive": positive,
                                  "negatives": negs}) + "\n")
            n_written += 1
    print(f"\nDONE. wrote {n_written} training examples -> train_triples.jsonl")
    print(f"cached embeddings -> {emb_path}")


if __name__ == "__main__":
    main()