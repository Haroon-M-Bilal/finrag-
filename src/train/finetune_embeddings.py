"""
STAGE 3b — Fine-tune the embedding model on finance.

Uses the mined triples (query, correct chunk, hard negative) with
MultipleNegativesRankingLoss: pull the query close to its correct chunk, push it
away from the hard negative AND from every other passage in the batch.

Saves the finance-tuned embedder to checkpoints/bge-small-finder.

Run:  python src\\train\\finetune_embeddings.py
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

PROC   = Path("data/finder/processed")
OUT    = Path("checkpoints/bge-small-finder")
EMBED  = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
EPOCHS, BATCH, LR = 3, 32, 2e-5
NEG_PER_EX = 2          # use the 2 hardest negatives per query
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"device: {DEVICE}")
    examples = []
    with open(PROC / "train_triples.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            anchor = PREFIX + d["query"]
            for neg in d["negatives"][:NEG_PER_EX]:
                # MNRL with an explicit hard negative: [anchor, positive, negative]
                examples.append(InputExample(texts=[anchor, d["positive"], neg]))
    print(f"training examples: {len(examples)}")

    model = SentenceTransformer(EMBED, device=DEVICE)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(loader) * EPOCHS * 0.1)

    print("fine-tuning...")
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=warmup,
        optimizer_params={"lr": LR},
        show_progress_bar=True,
        use_amp=True,                 # mixed precision -> fits 12GB, faster
    )
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    print(f"DONE. saved fine-tuned embedder -> {OUT}")


if __name__ == "__main__":
    main()