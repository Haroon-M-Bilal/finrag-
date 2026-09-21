"""
GENERATOR Step 1 (v5) - build Q&A fine-tuning data from FinDER.

WHY v5 EXISTS: THE v4 DATA TAUGHT REFUSAL

Measured on 40 held-out questions, the v4-trained Qwen-3B abstained on 39/40
with GOLD EVIDENCE IN CONTEXT, and 40/40 with retrieved context. The untrained
base model abstained on 7/40 and 9/40. Fine-tuning made the model strictly
worse at answering - it did not learn the domain, it learned to refuse.

Three design errors in v4 caused this:

  1. THE ABSTENTION TARGET WAS ONE IDENTICAL STRING, repeated across 606 of
     4,010 rows. A 3B model minimises loss most cheaply by memorising that
     string, not by learning when the context is insufficient.

  2. THE FOOTER WAS OVERWHELMINGLY A REFUSAL TOKEN. 3,519 qualitative rows
     ended `ANSWER: N/A` and 606 ended `ANSWER: INSUFFICIENT` against only 491
     carrying a real figure - 88% of targets taught "no value here". The
     unconditional footer was the right call; making its majority value a
     refusal was not.

  3. 15% ABSTENTION WAS TOO HIGH for the signal it needed to carry.

v5 CHANGES

  a. ABSTENTION RATE 15% -> 6%, and every abstention target is drawn from a
     pool of varied phrasings so there is no single string to memorise.
  b. QUALITATIVE FOOTERS ARE NO LONGER A REFUSAL TOKEN. A qualitative answer
     is not a failure to produce a number, so its footer is `ANSWER: TEXT`.
     `INSUFFICIENT` now means only one thing - the context genuinely lacks the
     answer - instead of sharing a semantic neighbourhood with 3,519 rows of
     "this question has no number".
  c. NUMERIC ROWS ARE UPWEIGHTED by duplication (NUMERIC_REPEAT), so questions
     carrying a real figure are not drowned by the qualitative majority. This
     is the cheapest available correction and it is reported, not hidden.
  d. Abstention examples are drawn preferentially from NUMERIC questions, so
     the model learns "the figure is not here" rather than "questions are
     unanswerable in general".

Everything else carries over from v4: the frozen filing-level split (the
previous positional holdout leaked filings between train and eval), same-filing
distractors mined with the fine-tuned retriever, grade-2-first gold ordering,
randomised gold position, and dropping numeric questions whose gold figure
could not be extracted with two independent signals rather than mislabelling
them N/A.

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

N_CTX = 3
ABSTAIN_FRAC = 0.06         # was 0.15
NUMERIC_REPEAT = 3          # upweight rows carrying a real figure
SEED = 42

SYSTEM = ("You are a financial analyst. Answer the question using only the "
          "provided context from SEC 10-K filings. If the context does not "
          "contain the information needed, say so explicitly. End every "
          "response with a line beginning 'ANSWER:'.")

# varied, so there is no single string to memorise
ABSTAIN_TEXTS = [
    "The provided context does not contain the information needed to answer "
    "this question.",
    "This cannot be answered from the excerpts supplied. The relevant figures "
    "are not present in the provided context.",
    "The context above does not include the data required for this question.",
    "I cannot answer this from the given context - the necessary disclosure "
    "does not appear in these excerpts.",
    "The supplied filing excerpts do not cover this. The information needed "
    "is not in the context provided.",
    "Based on the context given, this question cannot be answered; the "
    "relevant figures are absent.",
    "The excerpts provided do not contain the disclosure this question asks "
    "about.",
    "Answering this would require information that is not present in the "
    "context supplied.",
]


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
    return f"{SYSTEM}\n\nContext:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"


def main():
    rng = random.Random(SEED)

    corpus = {r["_id"]: r["text"] for r in jl(PROC / "corpus.jsonl")}
    answers = {r["_id"]: r for r in jl(PROC / "answers.jsonl")}
    numeric = {r["_id"]: r for r in jl(PROC / "answers_numeric.jsonl")}

    distract = {}
    rr = PROC / "train_triples_rr_split.jsonl"
    if rr.exists():
        for r in jl(rr):
            pool = distract.setdefault(str(r["qid"]), [])
            n_in = r.get("n_in_filing", len(r["negatives"]))
            pool.extend(r["negatives"][:n_in])
    print(f"distractor pools: {len(distract)} questions")

    stats = {}

    for name, out_name in (("train", "train"), ("test", "eval")):
        qs = jl(PROC / f"queries_{name}.jsonl")
        qr = load_qrels(PROC / f"qrels_{name}.tsv")
        rows = []
        c = {"total": 0, "numeric_trusted": 0, "numeric_dropped": 0,
             "qualitative": 0, "abstain": 0, "numeric_duplicated": 0}

        # decide abstention assignments up front, biased toward numeric
        eligible = [str(q["_id"]) for q in qs
                    if str(q["_id"]) in distract and str(q["_id"]) in qr]
        n_abstain = int(len(eligible) * ABSTAIN_FRAC) if name == "train" else 0
        num_elig = [q for q in eligible if numeric.get(q, {}).get("numeric")]
        oth_elig = [q for q in eligible if q not in set(num_elig)]
        rng.shuffle(num_elig)
        rng.shuffle(oth_elig)
        take_num = min(len(num_elig), int(n_abstain * 0.6))
        abstain_ids = set(num_elig[:take_num]
                          + oth_elig[:max(0, n_abstain - take_num)])

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
                continue

            if is_num and trusted:
                footer = nrec["answer_block"]
                c["numeric_trusted"] += 1
            else:
                footer = "TEXT"          # was N/A: a qualitative answer is not
                c["qualitative"] += 1    # a failure to produce a number

            gold_ids = sorted(gold_map, key=lambda x: -gold_map[x])
            gold_txt = [corpus[x] for x in gold_ids if x in corpus]
            if not gold_txt:
                continue
            pool = [d for d in distract.get(qid, []) if d]

            if qid in abstain_ids and pool:
                chunks = rng.sample(pool, min(N_CTX, len(pool)))
                target = (rng.choice(ABSTAIN_TEXTS)
                          + "\nANSWER: INSUFFICIENT")
                reps = 1
                c["abstain"] += 1
            else:
                keep = gold_txt[:max(1, N_CTX - 1)]
                fill = (rng.sample(pool, min(N_CTX - len(keep), len(pool)))
                        if pool else [])
                chunks = keep + fill
                rng.shuffle(chunks)
                target = f"{ans_text}\nANSWER: {footer}"
                reps = (NUMERIC_REPEAT if (is_num and trusted
                                           and name == "train") else 1)
                if reps > 1:
                    c["numeric_duplicated"] += reps - 1

            row = {
                "qid": qid,
                "question": q["text"],
                "chunk_texts": chunks,
                "prompt": build_prompt(q["text"], chunks),
                "target": target,
                "footer": ("INSUFFICIENT" if qid in abstain_ids and pool
                           else footer),
                "numeric": is_num and trusted and qid not in abstain_ids,
                "has_gold": qid not in abstain_ids,
                "n_chunks": len(chunks),
                "type": nrec.get("type", ""),
            }
            for _ in range(reps):
                rows.append(row)
            c["total"] += reps

        rng.shuffle(rows)
        out = PROC / f"qa_sft_{out_name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        foot = {}
        for r in rows:
            k = ("INSUFFICIENT" if r["footer"] == "INSUFFICIENT"
                 else "TEXT" if r["footer"] == "TEXT" else "value")
            foot[k] = foot.get(k, 0) + 1
        c["footer_distribution"] = foot
        stats[name] = c

        print(f"\n{out_name}: {c['total']} rows -> {out}")
        for k, v in c.items():
            if k != "total":
                print(f"    {k:<20} {v}")
        tot = sum(foot.values())
        print(f"    refusal footers      "
              f"{100 * foot.get('INSUFFICIENT', 0) / max(tot, 1):.1f}%  "
              f"(v4 was 88%)")

    stats["config"] = {
        "n_ctx": N_CTX, "abstain_frac": ABSTAIN_FRAC,
        "numeric_repeat": NUMERIC_REPEAT, "seed": SEED,
        "abstain_variants": len(ABSTAIN_TEXTS),
        "qualitative_footer": "TEXT (was N/A)",
        "note": "v4 data produced a model that abstained on 39/40 questions "
                "with gold context; base model abstained on 7/40",
    }
    json.dump(stats, open(PROC / "qa_sft_stats.json", "w"), indent=2)
    print(f"\nsaved -> {PROC / 'qa_sft_stats.json'}")


if __name__ == "__main__":
    main()