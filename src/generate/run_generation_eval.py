"""
GENERATOR Step 3 — end-to-end answer-quality eval (base Qwen vs QLoRA Qwen).

Both models answer the 300 held-out questions using the SAME retrieval
(fine-tuned embedder, top-3 chunks). Scored vs gold answers with:
  ROUGE-L (overlap), BLEU (n-gram), BERTScore-F1 (semantic, best for short answers).

Run:  python src\\generate\\run_generation_eval.py
Output: results\\generation.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROC = Path("data/finder/processed"); CKPT = Path("checkpoints")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = str(CKPT / "qwen-finder-qlora")
PREFIX = "Represent this sentence for searching relevant passages: "
TOPK_CTX = 3
SYS = "You are a financial analyst. Answer the question using ONLY the context. Be concise and numeric."
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    answers = {json.loads(l)["_id"]: json.loads(l)["answer"]
               for l in open(PROC / "answers.jsonl", encoding="utf-8")}
    qtext = {str(q["_id"]): q["text"] for q in jl(PROC / "queries.jsonl")}
    eval_ids = json.load(open(PROC / "qa_eval_ids.json"))
    eval_ids = [q for q in eval_ids if answers.get(q, "").strip()]
    print(f"eval questions: {len(eval_ids)}  device={DEV}")

    # ---- retrieve context with fine-tuned embedder ----
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
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)

    def gen_all(model, tag):
        out = []
        for k, qid in enumerate(eval_ids):
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": f"Context:\n{contexts[qid]}\n\nQuestion: {qtext[qid]}"}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=128, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            out.append(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip())
            if k % 50 == 0: print(f"  {tag} {k}/{len(eval_ids)}", end="\r")
        print()
        return out

    print("loading base Qwen (4-bit)...")
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    base.config.use_cache = True
    base_preds = gen_all(base, "base")

    print("attaching QLoRA adapters...")
    ft = PeftModel.from_pretrained(base, ADAPTER)
    ft_preds = gen_all(ft, "ft")
    del base, ft; torch.cuda.empty_cache()

    golds = [answers[q] for q in eval_ids]

    # ---- metrics ----
    from rouge_score import rouge_scorer
    import sacrebleu
    from bert_score import score as bertscore
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def metrics(preds):
        rouge = np.mean([rs.score(g, p)["rougeL"].fmeasure for g, p in zip(golds, preds)])
        bleu = np.mean([sacrebleu.sentence_bleu(p, [g]).score for g, p in zip(golds, preds)])
        _, _, F = bertscore(preds, golds, lang="en", verbose=False)
        return {"ROUGE-L": rouge, "BLEU": bleu, "BERTScore-F1": float(F.mean())}

    mb, mf = metrics(base_preds), metrics(ft_preds)
    cols = ["ROUGE-L", "BLEU", "BERTScore-F1"]
    rows = ["| System | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    rows.append("| Qwen2.5-7B (base) | " + " | ".join(f"{mb[c]:.4f}" for c in cols) + " |")
    rows.append("| Qwen2.5-7B + QLoRA (ours) | " + " | ".join(f"{mf[c]:.4f}" for c in cols) + " |")
    out = "# Generation quality (300 held-out, ft_dense retrieval)\n\n" + "\n".join(rows) + "\n"
    print("\n" + out)
    (RESULTS / "generation.md").write_text(out, encoding="utf-8")
    json.dump({"base": mb, "qlora": mf}, open(RESULTS / "generation.json", "w"), indent=2)
    # save a few examples for the demo / paper appendix
    ex = [{"q": qtext[q], "gold": answers[q], "base": base_preds[i], "qlora": ft_preds[i]}
          for i, q in enumerate(eval_ids[:10])]
    json.dump(ex, open(RESULTS / "generation_examples.json", "w"), indent=2)
    print("saved -> results/generation.md")


if __name__ == "__main__":
    main()