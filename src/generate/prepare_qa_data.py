"""
GENERATOR Step 1 — build Q&A fine-tuning data from FinDER.

Each training row teaches: given the retrieved context + question -> produce the
gold answer (FinDER's terse, numeric style). Uses the GOLD chunk as context for
clean supervision. Holds out the last 300 questions for end-to-end eval (those
are NEVER trained on; eval will use RETRIEVED context, not gold).

Run:  python src\\generate\\prepare_qa_data.py
Output:
    data/finder/processed/qa_sft.jsonl      # {question, context, answer}  (train)
    data/finder/processed/qa_eval_ids.json  # held-out question ids (eval)
"""
import json
from pathlib import Path

PROC = Path("data/finder/processed")
HOLDOUT = 300

corpus = {json.loads(l)["_id"]: json.loads(l)["text"]
          for l in open(PROC / "corpus.jsonl", encoding="utf-8")}
answers = {json.loads(l)["_id"]: json.loads(l) for l in open(PROC / "answers.jsonl", encoding="utf-8")}
queries = [json.loads(l) for l in open(PROC / "queries.jsonl", encoding="utf-8")]

qr = {}
f = open(PROC / "qrels.tsv", encoding="utf-8"); next(f)
for line in f:
    a, b, c = line.rstrip("\n").split("\t"); qr.setdefault(a, []).append(b)

scored = [q for q in queries if str(q["_id"]) in qr and str(q["_id"]) in answers]
eval_ids = [str(q["_id"]) for q in scored[-HOLDOUT:]]
train = scored[:-HOLDOUT]

n = 0
with open(PROC / "qa_sft.jsonl", "w", encoding="utf-8") as out:
    for q in train:
        qid = str(q["_id"])
        ans = answers[qid]["answer"]
        if not ans or not str(ans).strip():
            continue
        context = corpus[qr[qid][0]]            # gold chunk as context
        out.write(json.dumps({"question": q["text"], "context": context,
                              "answer": str(ans)}) + "\n")
        n += 1

json.dump(eval_ids, open(PROC / "qa_eval_ids.json", "w"))
print(f"DONE. train rows: {n}  | held-out eval questions: {len(eval_ids)}")