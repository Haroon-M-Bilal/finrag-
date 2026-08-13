"""
Fine-tune the reranker on the TRAIN SPLIT (leakage-free).

Trains the cross-encoder on (query, positive) -> 1 and (query, hard negative) -> 0,
with negatives mined from the split-trained retriever. Ends with a convergence
check on a held-out slice: if positives do not clearly outscore negatives, the
run did not learn and should not be trusted.

Saves to checkpoints/bge-reranker-finder-split.

Run:  python src\\train\\finetune_reranker_split.py
"""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

PROC = Path("data/finder/processed")
OUT = Path("checkpoints/bge-reranker-finder-split")
BASE = "BAAI/bge-reranker-base"
EPOCHS, BATCH, LR = 3, 16, 2e-5
NEG_PER_EX = 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)


def main():
    print(f"device: {DEV}")
    rows = [json.loads(l) for l in open(PROC / "train_triples_rr_split.jsonl", encoding="utf-8")]
    random.shuffle(rows)
    dev, train = rows[:200], rows[200:]

    examples = []
    for d in train:
        examples.append(InputExample(texts=[d["query"], d["positive"]], label=1.0))
        for neg in d["negatives"][:NEG_PER_EX]:
            examples.append(InputExample(texts=[d["query"], neg], label=0.0))
    random.shuffle(examples)
    print(f"training pairs: {len(examples)}  (dev held out: {len(dev)})")

    model = CrossEncoder(BASE, num_labels=1, device=DEV, max_length=512)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH, drop_last=True)
    warmup = int(len(loader) * EPOCHS * 0.1)

    print("fine-tuning reranker (train split)...")
    model.fit(train_dataloader=loader, epochs=EPOCHS, warmup_steps=warmup,
              optimizer_params={"lr": LR}, use_amp=True, show_progress_bar=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    print(f"saved -> {OUT}")

    print("\nCONVERGENCE CHECK (held-out dev):")
    ps = model.predict([[d["query"], d["positive"]] for d in dev], batch_size=64)
    ns = model.predict([[d["query"], d["negatives"][0]] for d in dev], batch_size=64)
    acc = float(np.mean(np.array(ps) > np.array(ns)))
    print(f"  mean positive: {np.mean(ps):.4f}   mean negative: {np.mean(ns):.4f}")
    print(f"  positive>negative accuracy: {acc:.3f}")
    print("  => LEARNED" if acc > 0.75 else "  => DID NOT LEARN - do not trust")


if __name__ == "__main__":
    main()