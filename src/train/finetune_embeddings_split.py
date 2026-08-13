"""
Fine-tune the embedder on the TRAIN SPLIT only (leakage-free).

MultipleNegativesRankingLoss with mined hard negatives. Training pairs come from
train-split filings only, so evaluation on test filings measures generalisation
rather than memorisation.

CHANGES FROM THE PREVIOUS VERSION:

  1. ONE EXAMPLE PER TRIPLE, MULTIPLE NEGATIVES. The old version emitted
     NEG_PER_EX separate InputExamples sharing the same anchor and positive.
     MNRL treats every other positive in the batch as a negative, so two copies
     of the same (anchor, positive) landing in one batch each became a false
     negative for the other. MNRL accepts [anchor, positive, neg1, ..., negN]
     directly, which uses the same negatives without the collision.

  2. HELD-OUT VALIDATION. A slice of the mined triples is held out and scored
     with TripletEvaluator (fraction of cases where the positive is closer to
     the anchor than the negative). The previous run reported no validation
     signal for the embedder at all, so there was no way to see overfitting.
     The held-out slice is split by QUESTION ID, not by row, since one question
     can produce several triples.

  3. READS processed_v4, and writes a train_config.json next to the checkpoint
     so the run is reconstructable.

Saves to checkpoints/bge-small-finder-v4.
Run:  python -u src\\train\\finetune_embeddings_split.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

PROC = Path("data/finder/processed_v4")
OUT = Path("checkpoints/bge-small-finder-v4")
EMBED = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "

EPOCHS, BATCH, LR = 3, 16, 2e-5
NEG_PER_EX = 4        # negatives packed into each example
VAL_FRAC = 0.05       # held-out questions for the triplet evaluator
SEED = 42

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def acc(evaluator, model) -> float:
    """TripletEvaluator returns a dict in sentence-transformers 3.x.
    Pin cosine explicitly: dot_accuracy uses an inverted sign convention
    and reads as 1 - cosine_accuracy."""
    r = evaluator(model)
    if not isinstance(r, dict):
        return float(r)
    for k, v in r.items():
        if k.endswith("cosine_accuracy"):
            return float(v)
    raise KeyError(f"no cosine_accuracy in {list(r)}")

def main():
    print(f"device: {DEV}")

    rows = [json.loads(l) for l in
            open(PROC / "train_triples_split.jsonl", encoding="utf-8")]
    if not rows:
        raise SystemExit("no triples found - run mine_negatives_split.py first")

    qids = sorted({r["qid"] for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * VAL_FRAC))
    val_qids = set(qids[:n_val])
    print(f"triples: {len(rows)}  questions: {len(qids)}  "
          f"held out: {len(val_qids)}")

    train_ex, val_a, val_p, val_n = [], [], [], []
    for r in rows:
        anchor = PREFIX + r["query"]
        negs = r["negatives"][:NEG_PER_EX]
        if not negs:
            continue
        if r["qid"] in val_qids:
            val_a.append(anchor)
            val_p.append(r["positive"])
            val_n.append(negs[0])
        else:
            train_ex.append(InputExample(texts=[anchor, r["positive"], *negs]))

    print(f"training examples: {len(train_ex)}  validation triples: {len(val_a)}")

    model = SentenceTransformer(EMBED, device=DEV)
    loader = DataLoader(train_ex, shuffle=True, batch_size=BATCH, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = TripletEvaluator(anchors=val_a, positives=val_p, negatives=val_n,
                                 name="finder-val", show_progress_bar=False)
    base_acc = acc(evaluator, model)
    print(f"baseline triplet accuracy (before training): {base_acc:.4f}")

    warmup = int(len(loader) * EPOCHS * 0.1)
    print("fine-tuning embedder (train split)...")
    model.fit(train_objectives=[(loader, loss)],
              evaluator=evaluator,
              evaluation_steps=max(1, len(loader) // 2),
              epochs=EPOCHS,
              warmup_steps=warmup,
              optimizer_params={"lr": LR},
              show_progress_bar=True,
              use_amp=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))

    final = acc(evaluator, model)
    print(f"final triplet accuracy: {final:.4f}")

    json.dump({
        "base_model": EMBED,
        "source": str(PROC),
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "lr": LR,
        "neg_per_example": NEG_PER_EX,
        "loss": "MultipleNegativesRankingLoss",
        "prefix_applied_to": "queries only",
        "val_frac": VAL_FRAC,
        "seed": SEED,
        "n_train_examples": len(train_ex),
        "n_val_triples": len(val_a),
        "baseline_triplet_accuracy": float(base_acc),
        "final_triplet_accuracy": float(final),
    }, open(OUT / "train_config.json", "w"), indent=2)

    print(f"DONE. saved -> {OUT}")


if __name__ == "__main__":
    main()