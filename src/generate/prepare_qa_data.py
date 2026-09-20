"""
GENERATOR Step 1 (v4) - build Q&A fine-tuning data from FinDER.

WHAT WAS WRONG WITH THE PREVIOUS VERSION

  1. LEAKAGE. It held out `scored[-300:]` - the last 300 questions by LIST
     POSITION. Questions about the same filing therefore sat on both sides, so
     the generator was evaluated on filings it had trained on. Same defect as
     the retriever's, one layer down. The split is now the frozen filing-level
     split: train questions come from train filings, evaluation from the 98
     held-out filings.

  2. TRAIN/INFERENCE MISMATCH. Training context was the gold chunk, always.
     Evaluation context is whatever retrieval returns, which frequently does
     not contain the answer. A model that has never seen a context lacking the
     answer has no reason to abstain, which is why it produced confident
     waffle. Training rows now mix gold with retrieved DISTRACTORS, and a
     deliberate fraction contain NO gold at all, with abstention as the target.
     (This is the RAFT recipe: train on the context distribution inference
     actually produces.)

  3. ARBITRARY GOLD CHUNK. `qr[qid][0]` took whichever chunk id happened to be
     first, ignoring grades. Context is now built grade-2 (primary evidence)
     first, then grade-1 supporting chunks.

  4. NO STRUCTURED TARGET. ROUGE and BERTScore cannot detect a wrong financial
     figure. Every target now ends with a machine-readable footer:

         ANSWER: 111.5 | million | USD     numeric question
         ANSWER: N/A                       qualitative question
         ANSWER: INSUFFICIENT              context does not contain the answer

     The footer is UNCONDITIONAL - it appears on every target regardless of
     question type. Teaching a 3B model WHEN to emit structure is a second task
     it will fail; teaching it to always end the same way is one rule, and the
     84.5% qualitative majority reinforces the habit rather than competing
     with it. Prose comes first and stays natural; the footer is stripped
     before ROUGE/BERTScore are computed.

  5. UNTRUSTED NUMERIC TARGETS ARE DROPPED, not silently labelled N/A.
     Labelling a numeric question N/A would teach the model to refuse
     calculation. Questions whose gold figure could not be extracted with two
     independent signals are excluded from training and counted.

  6. GOLD POSITION IS RANDOMISED among the context chunks. Always placing gold
     last teaches a positional shortcut and invites the "lost in the middle"
     failure at inference, where reranked order puts gold first.

DISTRACTORS come from `train_triples_rr_split.jsonl` - the same-filing hard
negatives already mined with the fine-tuned retriever. They are exactly what
routing + reranking will hand the generator at inference, so no new mining is
needed.

Run:  python -u src/generate/prepare_qa_data.py
Output: data/finder/processed_v4/qa_sft_train.jsonl
        data/finder/processed_v4/qa_sft_eval.jsonl
        data/finder/processed_v4/qa_sft_stats.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

PROC = Path("data/finder/processed_v4")

N_CTX = 3               # chunks per training context, matching inference top-3
ABSTAIN_FRAC = 0.15     # share of training rows with no gold in context
SEED = 42

SYSTEM = ("You are a financial analyst. Answer the question using only the "
          "provided context from SEC 10-K filings. If the context does not "
          "contain the information needed, say so explicitly. End every "
          "response with a line beginning 'ANSWER:'.")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_qrels(p):
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, s = line.rstrip("\n").split("\t")
            q.setdefault(a, {})[b] = int(s)
    return q


def build_prompt(question: str, chunks: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return (f"{SYSTEM}\n\nContext:\n{ctx}\n\nQuestion: {question}\n\nAnswer:")


def main():
    rng = random.Random(SEED)

    corpus = {r["_id"]: r["text"] for r in jl(PROC / "corpus.jsonl")}
    answers = {r["_id"]: r for r in jl(PROC / "answers.jsonl")}
    numeric = {r["_id"]: r for r in jl(PROC / "answers_numeric.jsonl")}

    splits = {}
    for name in ("train", "test"):
        qs = jl(PROC / f"queries_{name}.jsonl")
        qr = load_qrels(PROC / f"qrels_{name}.tsv")
        splits[name] = (qs, qr)

    # same-filing hard negatives, already mined with the FT retriever
    distract = {}
    rr_path = PROC / "train_triples_rr_split.jsonl"
    if rr_path.exists():
        for r in jl(rr_path):
            distract.setdefault(str(r["qid"]), [])
            for i, neg in enumerate(r["negatives"]):
                if i < r.get("n_in_filing", len(r["negatives"])):
                    distract[str(r["qid"])].append(neg)
    print(f"distractor pools available for {len(distract)} train questions")

    stats = {"train": {}, "test": {}}

    for name in ("train", "test"):
        qs, qr = splits[name]
        rows = []
        c = {"total": 0, "numeric_trusted": 0, "numeric_dropped": 0,
             "qualitative": 0, "abstain": 0, "no_distractors": 0}

        for q in qs:
            qid = str(q["_id"])
            gold_map = qr.get(qid, {})
            arec = answers.get(qid)
            if not gold_map or not arec:
                continue
            ans_text = str(arec.get("answer", "")).strip()
            if not ans_text:
                continue

            nrec = numeric.get(qid, {})
            is_num = bool(nrec.get("numeric"))
            trusted = nrec.get("tier") in ("anchored_verified", "single_figure")

            if is_num and not trusted:
                c["numeric_dropped"] += 1
                continue          # never label a numeric question N/A

            footer = nrec["answer_block"] if (is_num and trusted) else "N/A"
            if is_num:
                c["numeric_trusted"] += 1
            else:
                c["qualitative"] += 1

            # gold chunks, grade 2 first
            gold_ids = sorted(gold_map, key=lambda c_: -gold_map[c_])
            gold_txt = [corpus[c_] for c_ in gold_ids if c_ in corpus]
            if not gold_txt:
                continue

            pool = [d for d in distract.get(qid, []) if d]
            if not pool:
                c["no_distractors"] += 1

            abstain = (name == "train" and pool
                       and rng.random() < ABSTAIN_FRAC)

            if abstain:
                chunks = rng.sample(pool, min(N_CTX, len(pool)))
                target = ("The provided context does not contain the "
                          "information needed to answer this question.\n"
                          "ANSWER: INSUFFICIENT")
                c["abstain"] += 1
            else:
                keep_gold = gold_txt[:max(1, N_CTX - 1)]
                n_fill = max(0, N_CTX - len(keep_gold))
                fill = rng.sample(pool, min(n_fill, len(pool))) if pool else []
                chunks = keep_gold + fill
                rng.shuffle(chunks)          # gold position must not be fixed
                target = f"{ans_text}\nANSWER: {footer}"

            rows.append({
                "qid": qid,
                "question": q["text"],
                "chunk_texts": chunks,
                "prompt": build_prompt(q["text"], chunks),
                "target": target,
                "footer": "INSUFFICIENT" if abstain else footer,
                "numeric": is_num and not abstain,
                "has_gold": not abstain,
                "n_chunks": len(chunks),
                "type": nrec.get("type", ""),
            })
            c["total"] += 1

        out = PROC / f"qa_sft_{'train' if name == 'train' else 'eval'}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stats[name] = c
        print(f"{name}: {c['total']} rows -> {out}")
        for k, v in c.items():
            if k != "total":
                print(f"    {k:<18} {v}")

    stats["config"] = {"n_ctx": N_CTX, "abstain_frac": ABSTAIN_FRAC,
                       "seed": SEED, "source": str(PROC),
                       "footer": "unconditional; N/A | INSUFFICIENT | value",
                       "gold_position": "randomised",
                       "distractors": "same-filing hard negatives from the "
                                      "fine-tuned retriever"}
    json.dump(stats, open(PROC / "qa_sft_stats.json", "w"), indent=2)
    print(f"\nsaved -> {PROC / 'qa_sft_stats.json'}")


if __name__ == "__main__":
    main()