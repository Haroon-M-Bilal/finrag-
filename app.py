"""
FinRAG demo — live financial-report Q&A with your fine-tuned models.

Pipeline shown live: question -> fine-tuned embedder retrieves 10-K passages
-> QLoRA Qwen answers from them. Toggle "compare base model" to show the prof
the difference fine-tuning makes.

Run:  python app.py
Then open the http://127.0.0.1:7860 link it prints.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss, torch, gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROC = Path("data/finder/processed"); CKPT = Path("checkpoints")
BASE = "Qwen/Qwen2.5-7B-Instruct"; ADAPTER = str(CKPT / "qwen-finder-qlora")
PREFIX = "Represent this sentence for searching relevant passages: "
TOPK = 3
SYS = "You are a financial analyst. Answer the question using ONLY the context. Be concise and numeric."
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print("loading models (one minute)...")
corpus = [json.loads(l) for l in open(PROC / "corpus.jsonl", encoding="utf-8")]
cids = [c["_id"] for c in corpus]; ctexts = [c["text"] for c in corpus]
cid2text = dict(zip(cids, ctexts))
emb = SentenceTransformer(str(CKPT / "bge-small-finder"), device=DEV)
cemb = np.load(PROC / "corpus_emb_ft.npy")
index = faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
model = PeftModel.from_pretrained(base_model, ADAPTER)   # adapter on; disable for base
print("ready.")


def retrieve(q):
    qe = emb.encode([PREFIX + q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    _, idx = index.search(qe, TOPK)
    return [(cids[j].split("::")[0], cid2text[cids[j]]) for j in idx[0] if j != -1]


def generate(q, ctx, use_ft):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        if use_ft:                       # adapter active = our fine-tuned model
            o = model.generate(ids, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
        else:                            # adapter disabled = original base model
            with model.disable_adapter():
                o = model.generate(ids, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()


def answer(q, compare):
    if not q.strip():
        return "Enter a question.", "", ""
    hits = retrieve(q)
    ctx = "\n\n".join(t for _, t in hits)
    sources = "\n\n".join(f"**[{tic}]** {t[:400]}..." for tic, t in hits)
    ours = generate(q, ctx, use_ft=True)
    base = generate(q, ctx, use_ft=False) if compare else ""
    return ours, base, sources


with gr.Blocks(title="FinRAG — Financial Report Q&A") as demo:
    gr.Markdown("# FinRAG\nFine-tuned retrieval + QLoRA generation over S&P 500 10-K filings.")
    q = gr.Textbox(label="Financial question", placeholder="e.g. What was Apple's total net sales in fiscal 2023?")
    compare = gr.Checkbox(label="Compare against base (non-fine-tuned) model", value=True)
    btn = gr.Button("Answer", variant="primary")
    with gr.Row():
        out_ours = gr.Textbox(label="Our system (fine-tuned + QLoRA)")
        out_base = gr.Textbox(label="Base model (no fine-tuning)")
    src = gr.Markdown(label="Retrieved 10-K passages")
    btn.click(answer, [q, compare], [out_ours, out_base, src])

if __name__ == "__main__":
    demo.launch()