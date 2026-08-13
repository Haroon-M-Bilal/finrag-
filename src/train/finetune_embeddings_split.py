"""
Fine-tune the embedder on the TRAIN SPLIT only (leakage-free).

Identical method to the original run (MultipleNegativesRankingLoss with mined
hard negatives); the only change is that training pairs come from train-split
filings, so evaluation on test filings measures generalisation rather than
memorisation.

Saves to checkpoints/bge-small-finder-split.

Run:  python src\\train\\finetune_embeddings_split.py
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

PROC = Path("data/finder/processed")
OUT = Path("checkpoints/bge-small-finder-split")
EMBED = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
EPOCHS, BATCH, LR = 3, 32, 2e-5
NEG_PER_EX = 2
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"device: {DEV}")
    examples = []
    with open(PROC / "train_triples_split.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            anchor = PREFIX + d["query"]
            for neg in d["negatives"][:NEG_PER_EX]:
                examples.append(InputExample(texts=[anchor, d["positive"], neg]))
    print(f"training examples: {len(examples)}")

    model = SentenceTransformer(EMBED, device=DEV)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(loader) * EPOCHS * 0.1)

    print("fine-tuning embedder (train split)...")
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS, warmup_steps=warmup,
              optimizer_params={"lr": LR}, show_progress_bar=True, use_amp=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    print(f"DONE. saved -> {OUT}")


if __name__ == "__main__":
    main()