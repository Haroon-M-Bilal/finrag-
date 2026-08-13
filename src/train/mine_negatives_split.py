"""
Mine hard negatives from TRAIN-SPLIT questions only (leakage-free).

Uses the BASE embedder to mine negatives for the first embedder fine-tune.
Only questions in queries_train.jsonl are used, so no test-filing question ever
enters training.

CHANGES FROM THE PREVIOUS VERSION:

  1. POSITIVES USE GRADES. The old version took sorted(gold)[0], i.e. the
     alphabetically first chunk id - "AAPL::ch12" sorts before "AAPL::ch3", so
     the positive was arbitrary and often a grade-1 supporting fragment rather
     than the grade-2 primary. Now one triple is emitted per GRADE-2 chunk.

  2. EXCLUSION BAND ON NEGATIVES. The old filter was `cid not in gold`, which
     only excluded labelled positives. Chunks overlapping the evidence but just
     under the 80% union target - and the immediate neighbours of gold chunks,
     which share CHUNK_OVERLAP tokens with them by construction - were mined as
     hard negatives, training the model to push away text it should retrieve.
     Now excluded: gold, gold's adjacent chunks, and any candidate whose content
     overlap with the gold text exceeds EXCLUDE_OVERLAP.

  3. CONTROLLED IN-FILING / CROSS-FILING MIX. Under v4 labelling all gold sits
     in one filing, so globally-mined negatives are overwhelmingly other
     companies and the model can score well by matching the company name instead
     of learning relevance. In-filing negatives teach fine-grained relevance;
     cross-filing negatives teach document routing. The ratio is a reported
     constant, not an accident of the corpus.

  4. READS processed_v4.

  5. MANIFEST-GUARDED EMBEDDING CACHE. corpus_emb_base.npy is written with a
     sidecar recording model, shape and a hash of the chunk ids. A mismatched
     cache now hard-fails instead of silently indexing the wrong corpus.

Run:  python -u src\\train\\mine_negatives_split.py
Output: data/finder/processed_v4/train_triples_split.jsonl
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
EMBED = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "

TOP_N = 200            # global candidate pool per query
N_NEG_IN = 4           # hard negatives drawn from the question's own filing
N_NEG_CROSS = 4        # hard negatives drawn from other filings
EXCLUDE_OVERLAP = 0.35 # candidate is skipped if this fraction of its content
                       # tokens also appear in the gold text (near-miss guard)
ADJACENT = 1           # also exclude gold chunk index +/- this many

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOKRE = re.compile(r"[a-z0-9]{4,}")


def content(t: str) -> Counter:
    """Content-token counts. Length filter drops most function words cheaply."""
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
    npy = PROC / "corpus_emb_base.npy"
    meta = PROC / "corpus_emb_base.meta.json"
    fp = corpus_fingerprint(cids)

    if npy.exists() and meta.exists():
        m = json.load(open(meta, encoding="utf-8"))
        ok = (m.get("fingerprint") == fp
              and m.get("n_chunks") == len(cids)
              and m.get("model") == EMBED)
        if not ok:
            raise SystemExit(
                f"\nCACHE MISMATCH: {npy} was built for a different corpus.\n"
                f"  cached : {m.get('n_chunks')} chunks, model {m.get('model')}\n"
                f"  current: {len(cids)} chunks, model {EMBED}\n"
                f"Delete {npy} and {meta} and re-run.\n")
        print(f"loading cached corpus embeddings ({len(cids)} chunks, verified)...")
        return np.load(npy)

    if npy.exists():
        raise SystemExit(
            f"\n{npy} exists but has no manifest, so it cannot be verified "
            f"against the current corpus. Delete it and re-run.\n")

    print("encoding corpus with base BGE (no prefix on passages)...")
    cemb = model.encode(ctexts, batch_size=128, normalize_embeddings=True,
                        show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)
    np.save(npy, cemb)
    json.dump({"model": EMBED, "n_chunks": len(cids), "dim": int(cemb.shape[1]),
               "fingerprint": fp}, open(meta, "w"), indent=2)
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

    model = SentenceTransformer(EMBED, device=DEV)
    cemb = load_or_build_embeddings(model, cids, ctexts)

    index = faiss.IndexFlatIP(cemb.shape[1])
    index.add(cemb)

    qemb = model.encode([PREFIX + q["text"] for q in queries], batch_size=128,
                        normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)

    print("mining hard negatives (train split only)...")
    _, idxs = index.search(qemb, TOP_N)

    n_rows = 0
    n_excluded_band = 0
    n_short_in = 0
    n_short_cross = 0
    grade2_total = 0

    with open(PROC / "train_triples_split.jsonl", "w", encoding="utf-8") as out:
        for qi, q in enumerate(queries):
            qid = str(q["_id"])
            gold = qrels.get(qid, {})
            if not gold:
                continue

            tick = next(iter(gold)).split("::")[0]
            gold_idx = {cid2i[c] for c in gold if c in cid2i}

            # neighbours of gold share CHUNK_OVERLAP tokens by construction
            banned = set(gold_idx)
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
                nonlocal n_excluded_band
                if j in banned:
                    return False
                cc = content(ctexts[j])
                tot = sum(cc.values())
                if tot == 0:
                    return False
                if sum((cc & gold_content).values()) / tot >= EXCLUDE_OVERLAP:
                    n_excluded_band += 1
                    return False
                return True

            # cross-filing negatives from the global pool
            cross = [int(j) for j in idxs[qi]
                     if cids[int(j)].split("::")[0] != tick and keep(int(j))]
            cross = cross[:N_NEG_CROSS]
            if len(cross) < N_NEG_CROSS:
                n_short_cross += 1

            # in-filing negatives: rank the filing's own chunks directly, since
            # few of them survive into a global top-200
            fidx = [j for j in by_tick.get(tick, []) if j not in banned]
            if fidx:
                sims = cemb[fidx] @ qemb[qi]
                order = np.argsort(-sims)
                infile = []
                for o in order:
                    j = fidx[int(o)]
                    if keep(j):
                        infile.append(j)
                    if len(infile) >= N_NEG_IN:
                        break
            else:
                infile = []
            if len(infile) < N_NEG_IN:
                n_short_in += 1

            negs = [ctexts[j] for j in infile + cross]
            if not negs:
                continue

            # one row per grade-2 (primary) chunk
            primaries = [c for c, g in gold.items() if g == 2] or list(gold)
            grade2_total += len(primaries)
            for c in primaries:
                if c not in cid2i:
                    continue
                out.write(json.dumps({
                    "qid": qid,
                    "query": q["text"],
                    "pos_id": c,
                    "positive": ctexts[cid2i[c]],
                    "negatives": negs,
                    "n_in_filing": len(infile),
                    "n_cross_filing": len(cross),
                }) + "\n")
                n_rows += 1

    json.dump({
        "source": str(PROC),
        "embed_model": EMBED,
        "top_n": TOP_N,
        "n_neg_in_filing": N_NEG_IN,
        "n_neg_cross_filing": N_NEG_CROSS,
        "exclude_overlap": EXCLUDE_OVERLAP,
        "adjacent_excluded": ADJACENT,
        "n_train_questions": len(queries),
        "n_triples": n_rows,
        "n_primary_chunks": grade2_total,
        "candidates_dropped_by_band": n_excluded_band,
        "queries_short_of_in_filing_negs": n_short_in,
        "queries_short_of_cross_filing_negs": n_short_cross,
    }, open(PROC / "mining_stats.json", "w"), indent=2)

    print(f"\nDONE. wrote {n_rows} triples -> train_triples_split.jsonl")
    print(f"  primary (grade-2) chunks used : {grade2_total}")
    print(f"  candidates dropped by band    : {n_excluded_band}")
    print(f"  queries short of in-filing negs: {n_short_in}")
    print(f"  queries short of cross-filing  : {n_short_cross}")


if __name__ == "__main__":
    main()