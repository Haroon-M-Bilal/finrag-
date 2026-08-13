"""
Fine-tune the reranker on the TRAIN SPLIT (leakage-free).

Trains the cross-encoder on (query, positive) -> 1 and (query, hard negative) -> 0,
with negatives mined from the v4 fine-tuned retriever. Ends with a convergence
check on a held-out slice: if positives do not clearly outscore negatives, the
run did not learn and should not be trusted.

CHANGES FROM THE PREVIOUS VERSION:

  1. DEV SPLIT BY QUESTION, NOT BY ROW. The old version took rows[:200] as dev.
     4498 rows come from 4215 questions, so a question with two grade-2 chunks
     could land on both sides and the convergence check would be scoring
     questions the model had trained on.

  2. BASE-MODEL BASELINE. The untrained cross-encoder is scored on the same dev
     slice before training. This baseline is meaningful (unlike the embedder's
     triplet baseline, which was measured on negatives the base embedder had
     itself selected, and so was below chance by construction): these negatives
     were mined by the RETRIEVER, not by the reranker.

  3. READS processed_v4, trains from checkpoints/bge-small-finder-v4's mined
     negatives, and saves a train_config.json next to the checkpoint.

Saves to checkpoints/bge-reranker-finder-v4.
Run:  python -u src\\train\\finetune_reranker_split.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

PROC = Path("data/finder/processed_v4")
OUT = Path("checkpoints/bge-reranker-finder-v4")
BASE = "BAAI/bge-reranker-base"

EPOCHS, BATCH, LR = 3, 16, 2e-5
NEG_PER_EX = 3
DEV_QUESTIONS = 200
MAX_LEN = 512
SEED = 42

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def score_gap(model, dev_rows):
    """Mean positive score, mean negative score, and pairwise win rate."""
    ps = model.predict([[d["query"], d["positive"]] for d in dev_rows],
                       batch_size=64, show_progress_bar=False)
    ns = model.predict([[d["query"], d["negatives"][0]] for d in dev_rows],
                       batch_size=64, show_progress_bar=False)
    ps, ns = np.asarray(ps, dtype=float), np.asarray(ns, dtype=float)
    return float(ps.mean()), float(ns.mean()), float((ps > ns).mean())


def main():
    print(f"device: {DEV}")
    rng = random.Random(SEED)

    rows = [json.loads(l) for l in
            open(PROC / "train_triples_rr_split.jsonl", encoding="utf-8")]
    if not rows:
        raise SystemExit("no rows - run mine_negatives_rr_split.py first")

    # hold out whole QUESTIONS, so no question appears on both sides
    qids = sorted({r["qid"] for r in rows})
    rng.shuffle(qids)
    dev_qids = set(qids[:DEV_QUESTIONS])
    dev_rows = [r for r in rows if r["qid"] in dev_qids]
    train_rows = [r for r in rows if r["qid"] not in dev_qids]
    print(f"rows: {len(rows)}  questions: {len(qids)}  "
          f"dev questions: {len(dev_qids)} ({len(dev_rows)} rows)")

    examples = []
    for d in train_rows:
        examples.append(InputExample(texts=[d["query"], d["positive"]], label=1.0))
        for neg in d["negatives"][:NEG_PER_EX]:
            examples.append(InputExample(texts=[d["query"], neg], label=0.0))
    rng.shuffle(examples)
    n_pos = sum(1 for e in examples if e.label == 1.0)
    print(f"training pairs: {len(examples)}  "
          f"({n_pos} positive / {len(examples) - n_pos} negative)")

    model = CrossEncoder(BASE, num_labels=1, device=DEV, max_length=MAX_LEN)

    bp, bn, bacc = score_gap(model, dev_rows)
    print(f"\nBASE reranker on dev: pos {bp:.4f}  neg {bn:.4f}  "
          f"pos>neg {bacc:.3f}")

    loader = DataLoader(examples, shuffle=True, batch_size=BATCH, drop_last=True)
    warmup = int(len(loader) * EPOCHS * 0.1)
    print("\nfine-tuning reranker (train split)...")
    model.fit(train_dataloader=loader, epochs=EPOCHS, warmup_steps=warmup,
              optimizer_params={"lr": LR}, use_amp=True, show_progress_bar=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    print(f"saved -> {OUT}")

    fp, fn, facc = score_gap(model, dev_rows)
    print("\nCONVERGENCE CHECK (held-out dev questions):")
    print(f"  base : pos {bp:.4f}  neg {bn:.4f}  pos>neg {bacc:.3f}")
    print(f"  tuned: pos {fp:.4f}  neg {fn:.4f}  pos>neg {facc:.3f}")
    print("  => LEARNED" if facc > 0.75 else "  => DID NOT LEARN - do not trust")

    json.dump({
        "base_model": BASE,
        "source": str(PROC),
        "retriever_for_negatives": "checkpoints/bge-small-finder-v4",
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "lr": LR,
        "neg_per_example": NEG_PER_EX,
        "max_length": MAX_LEN,
        "seed": SEED,
        "n_train_pairs": len(examples),
        "n_dev_questions": len(dev_qids),
        "base_pos_gt_neg": bacc,
        "tuned_pos_gt_neg": facc,
        "base_mean_pos": bp, "base_mean_neg": bn,
        "tuned_mean_pos": fp, "tuned_mean_neg": fn,
    }, open(OUT / "train_config.json", "w"), indent=2)

    print(f"DONE. saved -> {OUT}")


if __name__ == "__main__":
    main()