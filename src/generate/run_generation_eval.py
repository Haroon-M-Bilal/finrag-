"""
GENERATION EVAL - multi-model, prediction-saving, extended metrics.

Evaluates base vs QLoRA generator on the 300 held-out questions using the SAME
fine-tuned retrieval for both.

Changes vs the previous version:
  - SAVES raw predictions to results/preds_<tag>.json, so any future metric can be
    recomputed instantly with score_only.py (no GPU, no re-generation)
  - BLEU reported on a 0-1 scale (sacrebleu returns 0-100; we divide by 100) so it
    is directly comparable to ROUGE and BERTScore in the same table
  - adds ROUGE-1 and ROUGE-2 alongside ROUGE-L, which are more informative than
    BLEU for long free-form answers

Usage:
    python src\generate\run_generation_eval.py Qwen/Qwen2.5-7B-Instruct
    python src\generate\run_generation_eval.py Qwen/Qwen2.5-3B-Instruct
    python src\generate\run_generation_eval.py mistralai/Mistral-7B-Instruct-v0.3
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROC = Path("data/finder/processed"); CKPT = Path("checkpoints")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
BASE = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
TAG = BASE.split("/")[-1].lower().replace(".", "-")
ADAPTER = str(CKPT / f"qlora-{TAG}")
PREFIX = "Represent this sentence for searching relevant passages: "
TOPK_CTX = 3
SYS = "You are a financial analyst. Answer the question using ONLY the context. Be concise and numeric."
DEV = "cuda" if torch.cuda.is_available() else "cpu"
METRIC_COLS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore-F1"]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def compute_metrics(preds, golds):
    """All metrics on a 0-1 scale."""
    from rouge_score import rouge_scorer
    import sacrebleu
    from bert_score import score as bertscore
    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for g, p in zip(golds, preds):
        s = rs.score(g, p)
        r1.append(s["rouge1"].fmeasure); r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    # sacrebleu returns 0-100 -> divide by 100 for a 0-1 scale
    bleu = float(np.mean([sacrebleu.sentence_bleu(p, [g]).score
                          for g, p in zip(golds, preds)])) / 100.0
    _, _, F = bertscore(preds, golds, lang="en", verbose=False)
    return {"ROUGE-1": float(np.mean(r1)), "ROUGE-2": float(np.mean(r2)),
            "ROUGE-L": float(np.mean(rl)), "BLEU": bleu,
            "BERTScore-F1": float(F.mean())}


def main():
    if not Path(ADAPTER).exists():
        raise SystemExit(f"adapter not found: {ADAPTER}\nTrain it first: finetune_qlora.py {BASE}")
    print(f"MODEL: {BASE}\nADAPTER: {ADAPTER}\ndevice: {DEV}")

    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    answers = {json.loads(l)["_id"]: json.loads(l)["answer"]
               for l in open(PROC / "answers.jsonl", encoding="utf-8")}
    qtext = {str(q["_id"]): q["text"] for q in jl(PROC / "queries.jsonl")}
    eval_ids = json.load(open(PROC / "qa_eval_ids.json"))
    eval_ids = [q for q in eval_ids if str(answers.get(q, "")).strip()]
    print(f"eval questions: {len(eval_ids)}")

    # ---- retrieval (identical for base and fine-tuned) ----
    emb = SentenceTransformer(str(CKPT / "bge-small-finder"), device=DEV)
    cemb = np.load(PROC / "corpus_emb_ft.npy")
    index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)
    qe = emb.encode([PREFIX + qtext[q] for q in eval_ids], normalize_embeddings=True,
                    convert_to_numpy=True).astype(np.float32)
    contexts = {}
    for i, qid in enumerate(eval_ids):
        _, idx = index.search(qe[i:i+1], TOPK_CTX)
        contexts[qid] = "\n\n".join(cid2text[cids[j]] for j in idx[0] if j != -1)
    del emb, index; torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print("loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                 device_map={"": 0})
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, ADAPTER)

    def gen_all(use_ft, tag):
        out = []
        for k, qid in enumerate(eval_ids):
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": f"Context:\n{contexts[qid]}\n\nQuestion: {qtext[qid]}"}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt").to(model.device)
            with torch.no_grad():
                if use_ft:
                    o = model.generate(ids, max_new_tokens=128, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                else:
                    with model.disable_adapter():
                        o = model.generate(ids, max_new_tokens=128, do_sample=False,
                                           pad_token_id=tok.pad_token_id)
            out.append(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip())
            if k % 50 == 0:
                print(f"  {tag} {k}/{len(eval_ids)}", end="\r")
        print()
        return out

    base_preds = gen_all(False, "base")
    ft_preds = gen_all(True, "ft")
    golds = [str(answers[q]) for q in eval_ids]
    del model; torch.cuda.empty_cache()

    # ---- SAVE PREDICTIONS (so metrics can be recomputed later, no GPU needed) ----
    json.dump({"model": BASE, "ids": eval_ids, "gold": golds,
               "base": base_preds, "qlora": ft_preds},
              open(RESULTS / f"preds_{TAG}.json", "w"), indent=2)
    print(f"predictions saved -> results/preds_{TAG}.json")

    mb, mf = compute_metrics(base_preds, golds), compute_metrics(ft_preds, golds)
    print(f"\n# {BASE} on {len(eval_ids)} held-out questions (all metrics 0-1)\n")
    print("| System | " + " | ".join(METRIC_COLS) + " |")
    print("|" + "---|" * (len(METRIC_COLS) + 1))
    print("| base | " + " | ".join(f"{mb[c]:.4f}" for c in METRIC_COLS) + " |")
    print("| + QLoRA | " + " | ".join(f"{mf[c]:.4f}" for c in METRIC_COLS) + " |")

    json.dump({"model": BASE, "base": mb, "qlora": mf},
              open(RESULTS / f"generation_{TAG}.json", "w"), indent=2)
    allp = RESULTS / "generation_all.json"
    allr = json.load(open(allp)) if allp.exists() else {}
    allr[BASE] = {"base": mb, "qlora": mf}
    json.dump(allr, open(allp, "w"), indent=2)
    print(f"\nsaved -> results/generation_{TAG}.json  (and generation_all.json)")


if __name__ == "__main__":
    main()