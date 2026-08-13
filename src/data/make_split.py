"""
FILING-LEVEL TRAIN/TEST SPLIT - eliminates evaluation leakage.

Problem this fixes: the embedder and reranker were trained on the same (query,
chunk) pairs later used for evaluation, so reported retrieval gains conflate
domain adaptation with memorisation.

Why split by FILING and not by question: chunks belong to a filing. If two
questions about AAPL land on different sides of the split, AAPL chunks appear in
both training and test and the leak persists. Splitting at the filing level makes
the test corpus genuinely unseen.

Output (data/finder/processed/):
    split_filings.json      {"train": [tickers...], "test": [tickers...]}
    qrels_train.tsv         relevance judgments for train filings only
    qrels_test.tsv          relevance judgments for test filings only
    queries_train.jsonl     questions whose gold chunk is in a train filing
    queries_test.jsonl      questions whose gold chunk is in a test filing

corpus.jsonl is unchanged: retrieval still searches ALL filings at test time
(that is the realistic setting). Only TRAINING pairs are restricted to train
filings.

Run:  python src\\data\\make_split.py
"""
from __future__ import annotations
import json, random
from pathlib import Path

PROC = Path("data/finder/processed")
TEST_FRAC = 0.2
SEED = 42
random.seed(SEED)


def load_qrels(p):
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, s = line.rstrip("\n").split("\t")
            q.setdefault(qid, []).append(cid)
    return q


def main():
    qrels = load_qrels(PROC / "qrels.tsv")
    queries = {str(json.loads(l)["_id"]): json.loads(l)["text"]
               for l in open(PROC / "queries.jsonl", encoding="utf-8")}

    # chunk ids look like "AAPL::ch12" -> filing ticker is the prefix
    filings = sorted({cid.split("::")[0] for cids in qrels.values() for cid in cids})
    random.shuffle(filings)
    n_test = max(1, int(len(filings) * TEST_FRAC))
    test_f = set(filings[:n_test]); train_f = set(filings[n_test:])
    print(f"filings: {len(filings)}  train: {len(train_f)}  test: {len(test_f)}")

    # a question goes to test only if ALL its gold chunks are in test filings
    # (questions spanning both sides are dropped, to keep the split clean)
    train_q, test_q, dropped = {}, {}, 0
    for qid, cids in qrels.items():
        fs = {c.split("::")[0] for c in cids}
        if fs <= train_f:
            train_q[qid] = cids
        elif fs <= test_f:
            test_q[qid] = cids
        else:
            dropped += 1
    print(f"questions -> train: {len(train_q)}  test: {len(test_q)}  dropped(span both): {dropped}")

    json.dump({"train": sorted(train_f), "test": sorted(test_f)},
              open(PROC / "split_filings.json", "w"), indent=2)

    for name, qd in (("train", train_q), ("test", test_q)):
        with open(PROC / f"qrels_{name}.tsv", "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for qid, cids in qd.items():
                for cid in cids:
                    f.write(f"{qid}\t{cid}\t1\n")
        with open(PROC / f"queries_{name}.jsonl", "w", encoding="utf-8") as f:
            for qid in qd:
                f.write(json.dumps({"_id": qid, "text": queries[qid]}) + "\n")

    print("\nDONE. wrote split_filings.json, qrels_{train,test}.tsv, queries_{train,test}.jsonl")
    print("Next: re-mine negatives and retrain using the TRAIN split only.")


if __name__ == "__main__":
    main()
    