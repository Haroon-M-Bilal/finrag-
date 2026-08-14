"""
Fine-tune the reranker on the TRAIN SPLIT (leakage-free).

CHANGES IN THIS VERSION

  1. PASSAGE TRUNCATION IS NOW EXPLICIT. Chunks were sized at 448 tokens using
     BGE's WordPiece tokenizer, but bge-reranker-base is XLM-RoBERTa and
     tokenizes the same text differently: measured mean 475, p90 525, so ~15% of
     chunks silently lost their tail inside CrossEncoder's internal truncation.
     Passages are now truncated deliberately to PASSAGE_TOKENS in the RERANKER's
     own tokenizer, identically at training and inference time, so the model
     always sees a complete unit rather than a variably-clipped one.

  2. DEV BREAKDOWN BY NEGATIVE TYPE. The previous run reported one pos>neg
     number, which hid the actual failure: the model separated cross-filing
     negatives easily and in-filing negatives barely at all. In-filing accuracy
     is now reported separately, and it is the number that matters, because
     under ticker-routed retrieval every candidate is already from the correct
     filing.

  3. Dev split by QUESTION, not row, and a base-model baseline on the same
     slice (negatives were mined by the retriever, not the reranker, so this
     baseline is meaningful).

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
from transformers import AutoTokenizer

PROC = Path("data/finder/processed_v4")
OUT = Path("checkpoints/bge-reranker-finder-v4")
BASE = "BAAI/bge-reranker-base"

EPOCHS, BATCH, LR = 3, 16, 2e-5
NEG_PER_EX = 4
DEV_QUESTIONS = 200
MAX_LEN = 512
PASSAGE_TOKENS = 460     # leaves room for query + specials inside MAX_LEN
SEED = 42

DEV = "cuda" if torch.cuda.is_available() else "cpu"

_tok = AutoTokenizer.from_pretrained(BASE)
_trunc_cache: dict[str, str] = {}


def clip(text: str) -> str:
    """Truncate to PASSAGE_TOKENS in the RERANKER's tokenizer, not BGE's."""
    hit = _trunc_cache.get(text)
    if hit is not None:
        return hit
    ids = _tok.encode(text, add_special_tokens=False)
    out = (_tok.decode(ids[:PASSAGE_TOKENS], skip_special_tokens=True)
           if len(ids) > PASSAGE_TOKENS else text)
    _trunc_cache[text] = out
    return out


def score_gap(model, rows, which="all"):
    """Mean positive score, mean negative score, pairwise win rate."""
    pairs_p, pairs_n = [], []
    for d in rows:
        negs = d["negatives"]
        if which == "in":
            negs = negs[:d.get("n_in_filing", 0)]
        elif which == "cross":
            negs = negs[d.get("n_in_filing", 0):]
        if not negs:
            continue
        pairs_p.append([d["query"], clip(d["positive"])])
        pairs_n.append([d["query"], clip(negs[0])])
    if not pairs_p:
        return float("nan"), float("nan"), float("nan"), 0
    ps = np.asarray(model.predict(pairs_p, batch_size=64,
                                 show_progress_bar=False), dtype=float)
    ns = np.asarray(model.predict(pairs_n, batch_size=64,
                                  show_progress_bar=False), dtype=float)
    return float(ps.mean()), float(ns.mean()), float((ps > ns).mean()), len(ps)


def report(model, rows, label):
    out = {}
    for which in ("all", "in", "cross"):
        p, n, a, k = score_gap(model, rows, which)
        out[which] = {"pos": p, "neg": n, "acc": a, "n": k}
        print(f"  {label:5s} {which:5s}: pos {p:.4f}  neg {n:.4f}  "
              f"pos>neg {a:.3f}  (n={k})")
    return out


def main():
    print(f"device: {DEV}")
    rng = random.Random(SEED)

    rows = [json.loads(l) for l in
            open(PROC / "train_triples_rr_split.jsonl", encoding="utf-8")]
    if not rows:
        raise SystemExit("no rows - run mine_negatives_rr_split.py first")

    qids = sorted({r["qid"] for r in rows})
    rng.shuffle(qids)
    dev_qids = set(qids[:DEV_QUESTIONS])
    dev_rows = [r for r in rows if r["qid"] in dev_qids]
    train_rows = [r for r in rows if r["qid"] not in dev_qids]
    print(f"rows: {len(rows)}  questions: {len(qids)}  "
          f"dev questions: {len(dev_qids)} ({len(dev_rows)} rows)")

    frac_in = np.mean([r.get("n_in_filing", 0) / max(len(r["negatives"]), 1)
                       for r in rows])
    print(f"in-filing negative fraction in training data: {frac_in:.3f}")

    examples = []
    for d in train_rows:
        examples.append(InputExample(texts=[d["query"], clip(d["positive"])],
                                     label=1.0))
        for neg in d["negatives"][:NEG_PER_EX]:
            examples.append(InputExample(texts=[d["query"], clip(neg)],
                                         label=0.0))
    rng.shuffle(examples)
    n_pos = sum(1 for e in examples if e.label == 1.0)
    print(f"training pairs: {len(examples)}  "
          f"({n_pos} positive / {len(examples) - n_pos} negative)")
    print(f"passages clipped to {PASSAGE_TOKENS} XLM-R tokens; "
          f"cache size {len(_trunc_cache)}")

    model = CrossEncoder(BASE, num_labels=1, device=DEV, max_length=MAX_LEN)

    print("\nBASE reranker on dev:")
    base_stats = report(model, dev_rows, "base")

    loader = DataLoader(examples, shuffle=True, batch_size=BATCH, drop_last=True)
    warmup = int(len(loader) * EPOCHS * 0.1)
    print("\nfine-tuning reranker...")
    model.fit(train_dataloader=loader, epochs=EPOCHS, warmup_steps=warmup,
              optimizer_params={"lr": LR}, use_amp=True, show_progress_bar=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))

    print("\nTUNED reranker on dev:")
    tuned_stats = report(model, dev_rows, "tuned")

    in_acc = tuned_stats["in"]["acc"]
    print(f"\nIN-FILING accuracy is the number that matters: {in_acc:.3f}")
    print("  => USABLE" if in_acc > 0.70 else
          "  => STILL WEAK on same-company discrimination")

    json.dump({
        "base_model": BASE, "source": str(PROC),
        "retriever_for_negatives": "checkpoints/bge-small-finder-v4",
        "epochs": EPOCHS, "batch_size": BATCH, "lr": LR,
        "neg_per_example": NEG_PER_EX, "max_length": MAX_LEN,
        "passage_tokens": PASSAGE_TOKENS,
        "passage_tokenizer": BASE,
        "in_filing_negative_frac": float(frac_in),
        "seed": SEED, "n_train_pairs": len(examples),
        "n_dev_questions": len(dev_qids),
        "base": base_stats, "tuned": tuned_stats,
    }, open(OUT / "train_config.json", "w"), indent=2)

    print(f"DONE. saved -> {OUT}")


if __name__ == "__main__":
    main()