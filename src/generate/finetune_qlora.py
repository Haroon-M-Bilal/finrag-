"""
GENERATOR Step 2 — QLoRA fine-tune Qwen2.5-7B-Instruct on FinDER Q&A.

4-bit base + LoRA adapters (fits 12GB). Loss is computed ONLY on the answer
tokens (the prompt/context is masked), so the model learns to ANSWER, not to
parrot the context. Saves LoRA adapters to checkpoints/qwen-finder-qlora.

If you hit CUDA out-of-memory: set MODEL to the 3B line below and/or lower MAXLEN.

Run:  python src\\generate\\finetune_qlora.py
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

PROC  = Path("data/finder/processed")
OUT   = Path("checkpoints/qwen-finder-qlora")
MODEL = "Qwen/Qwen2.5-7B-Instruct"        # OOM? -> "Qwen/Qwen2.5-3B-Instruct"
MAXLEN = 1024
EPOCHS, LR = 1, 2e-4
SYS = "You are a financial analyst. Answer the question using ONLY the context. Be concise and numeric."


def build(tok, ex):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}"}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    ans_ids = tok(ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + ans_ids)[:MAXLEN]
    labels = ([-100] * len(prompt_ids) + ans_ids)[:MAXLEN]
    return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, tok): self.tok = tok
    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = self.tok.pad_token_id
        ii, ll, am = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad] * n)
            ll.append(b["labels"] + [-100] * n)
            am.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}


def main():
    print("loading tokenizer + 4-bit model (first run downloads ~5GB)...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map={"": 0})
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(PROC / "qa_sft.jsonl", encoding="utf-8")]
    data = [build(tok, r) for r in rows]
    print(f"training examples: {len(data)}")

    args = TrainingArguments(
        output_dir="checkpoints/_qlora_tmp",
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS, learning_rate=LR, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", logging_steps=20, save_strategy="no",
        warmup_ratio=0.05, report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=data,
            data_collator=Collator(tok)).train()
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    print(f"DONE. saved LoRA adapters -> {OUT}")


if __name__ == "__main__":
    main()