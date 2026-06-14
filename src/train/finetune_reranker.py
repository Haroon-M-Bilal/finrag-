"""
STAGE 3c (v2) — Fine-tune the reranker, fixed.

Fixes the collapse from v1:
  - balanced sampling (1 positive : 3 negatives, was 1:4)
  - higher LR (2e-5) and 3 epochs so it actually learns
  - holds out a small dev set and prints a CONVERGENCE CHECK at the end:
    mean positive score vs mean negative score + pairwise accuracy.
    If positives don't clearly outscore negatives, it did NOT learn.

Saves to checkpoints/bge-reranker-finder (overwrites the broken one).

Run:  python src\\train\\finetune_reranker.py
"""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

PROC   = Path("data/finder/processed")
OUT    = Path("checkpoints/bge-reranker-finder")
BASE   = "BAAI/bge-reranker-base"
EPOCHS, BATCH, LR = 3, 16, 2e-5
NEG_PER_EX = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)


def main():
    print(f"device: {DEVICE}")
    rows = [json.loads(l) for l in open(PROC / "train_triples_rr.jsonl", encoding="utf-8")]
    random.shuffle(rows)
    dev = rows[:200]; train = rows[200:]      # hold out 200 for the convergence check

    examples = []
    for d in train:
        examples.append(InputExample(texts=[d["query"], d["positive"]], label=1.0))
        for neg in d["negatives"][:NEG_PER_EX]:
            examples.append(InputExample(texts=[d["query"], neg], label=0.0))
    random.shuffle(examples)
    print(f"training pairs: {len(examples)}  (dev held out: {len(dev)})")

    model = CrossEncoder(BASE, num_labels=1, device=DEVICE, max_length=512)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH, drop_last=True)
    warmup = int(len(loader) * EPOCHS * 0.1)

    print("fine-tuning reranker...")
    model.fit(train_dataloader=loader, epochs=EPOCHS, warmup_steps=warmup,
              optimizer_params={"lr": LR}, use_amp=True, show_progress_bar=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    print(f"saved -> {OUT}")

    # ---- convergence check on held-out dev ----
    print("\nCONVERGENCE CHECK (held-out dev):")
    pos_pairs = [[d["query"], d["positive"]] for d in dev]
    neg_pairs = [[d["query"], d["negatives"][0]] for d in dev]
    ps = model.predict(pos_pairs, batch_size=64)
    ns = model.predict(neg_pairs, batch_size=64)
    acc = float(np.mean(np.array(ps) > np.array(ns)))
    print(f"  mean positive score: {np.mean(ps):.4f}")
    print(f"  mean negative score: {np.mean(ns):.4f}")
    print(f"  positive>negative accuracy: {acc:.3f}")
    if acc > 0.80 and np.mean(ps) > np.mean(ns):
        print("  => LEARNED CORRECTLY. proceed to eval.")
    else:
        print("  => DID NOT LEARN. do not trust; tell Claude this number.")


if __name__ == "__main__":
    main()