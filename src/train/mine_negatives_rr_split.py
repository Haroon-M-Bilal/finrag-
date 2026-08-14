"""
Mine reranker negatives from the v4 fine-tuned retriever (train questions only).

WHY THIS WAS REWRITTEN

The first version deliberately did NOT force an in-filing / cross-filing ratio,
on the reasoning that the reranker only ever sees what the retriever hands it,
so its training distribution should be that pool as it is. Measured, that pool
was 11.3% in-filing: 89% of its negatives were chunks from OTHER COMPANIES.

The reranker therefore learned "is this the right company?" rather than "does
this passage answer the question?". Two pieces of evidence in the v4 ablation:

  - the UNTRAINED bge-reranker-base scored higher than the fine-tuned one
    (NDCG@10 0.1717 vs 0.1530)
  - applied on top of ticker-routed retrieval, where every candidate is already
    from the correct filing, the fine-tuned reranker destroyed the ranking
    (0.4388 -> 0.2024, paired bootstrap CI [-0.2914, -0.1813])

Company identity is exactly the signal ticker routing removes, so a reranker
trained on it has nothing left to contribute.

FIX: force a majority of in-filing negatives (4 in-filing / 2 cross-filing).
In-filing negatives are the hard case - same company, same boilerplate, wrong
passage - which is the discrimination a cross-encoder is supposed to add.
Cross-filing negatives are kept in the minority so the model still works in the
un-routed setting.

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

TOP_N = 100
N_NEG_IN = 4            # in-filing: same company, wrong passage (the hard case)
N_NEG_CROSS = 2         # cross-filing: keeps the un-routed setting working
EXCLUDE_OVERLAP = 0.50
ADJACENT = 1

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOKRE = re.compile(r"[a-z0-9]{4,}")


def content(t: str) -> Counter:
    return Counter(TOKRE.findall(t.lower()))


def load_qrels(p):
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
        if (m.get("fingerprint") != fp or m.get("n_chunks") != len(cids)
                or m.get("model") != EMBED_FT):
            raise SystemExit(f"\nCACHE MISMATCH: delete {npy} and {meta}.\n")
        print(f"loading cached FT corpus embeddings ({len(cids)} chunks, verified)...")
        return np.load(npy)
    if npy.exists():
        raise SystemExit(f"\n{npy} has no manifest. Delete it and re-run.\n")
    print("encoding corpus with the v4 fine-tuned embedder (~9 min)...")
    cemb = model.encode(ctexts, batch_size=128, normalize_embeddings=True,
                        show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)
    np.save(npy, cemb)
    json.dump({"model": EMBED_FT, "n_chunks": len(cids),
               "dim": int(cemb.shape[1]), "fingerprint": fp},
              open(meta, "w"), indent=2)
    return cemb


def main():
    corpus = [json.loads(l) for l in open(PROC / "corpus.jsonl", encoding="utf-8")]
    cids = [c["_id"] for c in corpus]
    ctexts = [c["text"] for c in corpus]
    cid2i = {c: i for i, c in enumerate(cids)}

    by_tick = {}
    for i, c in enumerate(cids):
        by_tick.setdefault(c.split("::")[0], []).append(i)

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

    print("mining negatives (in-filing majority)...")
    _, idxs = index.search(qemb, TOP_N)

    n_rows = n_band = n_short_in = n_short_cross = 0
    in_tot = neg_tot = gold_in_pool = 0

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

            def keep(j: int) -> bool:
                nonlocal n_band
                if j in banned:
                    return False
                cc = content(ctexts[j])
                tot = sum(cc.values())
                if tot == 0:
                    return False
                if sum((cc & gold_content).values()) / tot >= EXCLUDE_OVERLAP:
                    n_band += 1
                    return False
                return True

            pool = [int(j) for j in idxs[qi]]
            if any(cids[j] in gold for j in pool):
                gold_in_pool += 1

            # in-filing: rank the company's own chunks directly. these are the
            # hard negatives - same company, same boilerplate, wrong passage.
            fidx = [j for j in by_tick.get(tick, []) if j not in banned]
            infile = []
            if fidx:
                sims = cemb[fidx] @ qemb[qi]
                for o in np.argsort(-sims):
                    j = fidx[int(o)]
                    if keep(j):
                        infile.append(j)
                    if len(infile) >= N_NEG_IN:
                        break
            if len(infile) < N_NEG_IN:
                n_short_in += 1

            cross = []
            for j in pool:
                if cids[j].split("::")[0] == tick:
                    continue
                if keep(j):
                    cross.append(j)
                if len(cross) >= N_NEG_CROSS:
                    break
            if len(cross) < N_NEG_CROSS:
                n_short_cross += 1

            negs = infile + cross
            if not negs:
                continue
            in_tot += len(infile)
            neg_tot += len(negs)

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
                    "n_in_filing": len(infile),
                    "n_cross_filing": len(cross),
                }) + "\n")
                n_rows += 1

    json.dump({
        "source": str(PROC), "retriever": EMBED_FT, "top_n": TOP_N,
        "n_neg_in_filing": N_NEG_IN, "n_neg_cross_filing": N_NEG_CROSS,
        "exclude_overlap": EXCLUDE_OVERLAP, "adjacent_excluded": ADJACENT,
        "forced_filing_ratio": True,
        "n_train_questions": len(queries), "n_rows": n_rows,
        "candidates_dropped_by_band": n_band,
        "queries_short_of_in_filing": n_short_in,
        "queries_short_of_cross_filing": n_short_cross,
        "in_filing_negative_frac": round(in_tot / max(neg_tot, 1), 3),
        "gold_recall_at_top_n": round(gold_in_pool / max(len(queries), 1), 3),
    }, open(PROC / "mining_stats_rr.json", "w"), indent=2)

    print(f"\nDONE. wrote {n_rows} rows -> train_triples_rr_split.jsonl")
    print(f"  in-filing negative fraction: {in_tot / max(neg_tot, 1):.3f} "
          f"(was 0.113)")
    print(f"  candidates dropped by band : {n_band}")
    print(f"  short of in-filing negs    : {n_short_in}")
    print(f"  short of cross-filing negs : {n_short_cross}")
    print(f"  gold recall @ top-{TOP_N}      : "
          f"{gold_in_pool / max(len(queries), 1):.3f}")


if __name__ == "__main__":
    main()