"""
FILING-LEVEL TRAIN/TEST SPLIT - eliminates evaluation leakage.

Problem this fixes: the embedder and reranker were trained on the same (query,
chunk) pairs later used for evaluation, so reported retrieval gains conflate
domain adaptation with memorisation.

Why split by FILING and not by question: chunks belong to a filing. If two
questions about AAPL land on different sides of the split, AAPL chunks appear in
both training and test and the leak persists. Splitting at the filing level makes
the test corpus genuinely unseen.

CHANGES FROM THE PREVIOUS VERSION:
  1. Reads processed_v4 (filing-restricted, union-coverage qrels).
  2. GRADES ARE PRESERVED. The old version wrote a hard-coded score of 1 for
     every pair, silently collapsing graded qrels to binary and making NDCG a
     monotone function of MRR - the exact defect the graded labels were built
     to fix.
  3. The split is FROZEN. If split_filings.json exists it is loaded, not
     regenerated. random.seed(42) alone is not enough: the same seed over a
     different filing list yields a different split, so the embedder, reranker
     and generator could otherwise each train against a different partition.
     Delete the file deliberately to re-draw.

Output (data/finder/processed_v4/):
    split_filings.json      {"train": [tickers...], "test": [tickers...]}
    qrels_train.tsv         graded relevance judgments, train filings only
    qrels_test.tsv          graded relevance judgments, test filings only
    queries_train.jsonl     questions whose gold chunks are in train filings
    queries_test.jsonl      questions whose gold chunks are in test filings

corpus.jsonl is unchanged: retrieval still searches ALL filings at test time
(that is the realistic setting). Only TRAINING pairs are restricted to train
filings.

Run:  python -u src\\data\\make_split.py
"""
from __future__ import annotations

import json, random
from pathlib import Path

PROC = Path("data/finder/processed_v4")
TEST_FRAC = 0.2
SEED = 42


def load_qrels(p):
    """qid -> {cid: grade}. Grades are kept."""
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, s = line.rstrip("\n").split("\t")
            q.setdefault(qid, {})[cid] = int(s)
    return q


def main():
    qrels = load_qrels(PROC / "qrels.tsv")
    queries = {}
    for l in open(PROC / "queries.jsonl", encoding="utf-8"):
        r = json.loads(l)
        queries[str(r["_id"])] = r["text"]

    filings = sorted({cid.split("::")[0] for d in qrels.values() for cid in d})

    split_path = PROC / "split_filings.json"
    if split_path.exists():
        sp = json.load(open(split_path, encoding="utf-8"))
        train_f, test_f = set(sp["train"]), set(sp["test"])
        print(f"LOADED frozen split from {split_path}")
        missing = set(filings) - (train_f | test_f)
        extra = (train_f | test_f) - set(filings)
        if missing or extra:
            print(f"  WARNING: {len(missing)} filings in qrels but not in split, "
                  f"{len(extra)} in split but not in qrels")
            print("  The corpus changed since the split was drawn. Delete "
                  "split_filings.json to re-draw (this invalidates any model "
                  "already trained on the old split).")
    else:
        rng = random.Random(SEED)
        shuffled = list(filings)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * TEST_FRAC))
        test_f, train_f = set(shuffled[:n_test]), set(shuffled[n_test:])
        json.dump({"seed": SEED, "test_frac": TEST_FRAC,
                   "train": sorted(train_f), "test": sorted(test_f)},
                  open(split_path, "w"), indent=2)
        print(f"DREW new split, seed={SEED} -> {split_path}")

    print(f"filings: {len(filings)}  train: {len(train_f)}  test: {len(test_f)}")

    # a question goes to a side only if ALL its gold chunks are on that side.
    # under v4 labelling every question's gold sits in one filing, so nothing
    # should span - the counter is kept as a guard, not as a feature.
    train_q, test_q, spanning = {}, {}, 0
    for qid, d in qrels.items():
        fs = {c.split("::")[0] for c in d}
        if fs <= train_f:
            train_q[qid] = d
        elif fs <= test_f:
            test_q[qid] = d
        else:
            spanning += 1

    print(f"questions -> train: {len(train_q)}  test: {len(test_q)}  "
          f"spanning both: {spanning}")
    assert spanning == 0, (
        f"{spanning} questions have gold chunks in multiple filings - v4 "
        "labelling should make this impossible. Re-run prepare_finder_v4.py.")

    for name, qd in (("train", train_q), ("test", test_q)):
        with open(PROC / f"qrels_{name}.tsv", "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for qid, d in qd.items():
                for cid, g in d.items():
                    f.write(f"{qid}\t{cid}\t{g}\n")
        with open(PROC / f"queries_{name}.jsonl", "w", encoding="utf-8") as f:
            for qid in qd:
                f.write(json.dumps({"_id": qid, "text": queries[qid]}) + "\n")

    def grade_hist(qd):
        h = {}
        for d in qd.values():
            for g in d.values():
                h[g] = h.get(g, 0) + 1
        return dict(sorted(h.items()))

    print(f"\ngrade distribution  train: {grade_hist(train_q)}  "
          f"test: {grade_hist(test_q)}")
    print("DONE. wrote split_filings.json, qrels_{train,test}.tsv, "
          "queries_{train,test}.jsonl")
    print("Next: mine negatives and retrain using the TRAIN split only.")


if __name__ == "__main__":
    main()