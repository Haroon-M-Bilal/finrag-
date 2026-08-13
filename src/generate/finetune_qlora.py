"""
GENERATOR fine-tuning (QLoRA) - multi-model, checkpointed, auto-resuming.

Usage:
    python src\generate\finetune_qlora.py Qwen/Qwen2.5-3B-Instruct
    python src\generate\finetune_qlora.py Qwen/Qwen2.5-7B-Instruct
    python src\generate\finetune_qlora.py Qwen/Qwen2.5-14B-Instruct
    python src\generate\finetune_qlora.py mistralai/Mistral-7B-Instruct-v0.3

Features:
  - larger effective batch (grad-accum 16) + 2 epochs -> smooth, converging loss
  - held-out validation set; logs BOTH train and validation loss
  - saves loss history to results/qlora_loss_<tag>.json for plotting
  - CHECKPOINTS every 50 steps and AUTO-RESUMES if a run is interrupted
    (so a crash 10 hours into a 14B run costs minutes, not hours)
  - keeps only the 2 most recent checkpoints to limit disk use
"""
from __future__ import annotations
import json, sys, shutil
from pathlib import Path
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

PROC = Path("data/finder/processed")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
TAG = MODEL.split("/")[-1].lower().replace(".", "-")
OUT = Path(f"checkpoints/qlora-{TAG}")
CKPT_DIR = Path(f"checkpoints/_tmp_{TAG}")          # holds resumable checkpoints
# 14B is tight on 12 GB -> shorter sequences for that one only
MAXLEN = 768 if "14b" in TAG else 1024
EPOCHS, GRAD_ACCUM, LR = 2, 16, 2e-4
SAVE_EVERY = 50                                      # checkpoint frequency (steps)
VAL_N = 200
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
        m = max(len(b["input_ids"]) for b in batch); pad = self.tok.pad_token_id
        ii, ll, am = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad] * n)
            ll.append(b["labels"] + [-100] * n)
            am.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}


def latest_checkpoint(d: Path):
    if not d.exists():
        return None
    cks = [p for p in d.glob("checkpoint-*") if p.is_dir()]
    if not cks:
        return None
    return max(cks, key=lambda p: int(p.name.split("-")[-1]))


def main():
    print(f"MODEL: {MODEL}  ->  {OUT}   (maxlen={MAXLEN})")
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
    val_rows, train_rows = rows[-VAL_N:], rows[:-VAL_N]
    train_data = [build(tok, r) for r in train_rows]
    val_data = [build(tok, r) for r in val_rows]
    print(f"train: {len(train_data)}  val: {len(val_data)}")

    args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS, learning_rate=LR, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", warmup_ratio=0.05, report_to="none",
        logging_steps=10, eval_strategy="steps", eval_steps=25,
        per_device_eval_batch_size=1,
        save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data,
                      eval_dataset=val_data, data_collator=Collator(tok))

    resume = latest_checkpoint(CKPT_DIR)
    if resume:
        print(f"RESUMING from {resume}")
        trainer.train(resume_from_checkpoint=str(resume))
    else:
        trainer.train()

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT)); tok.save_pretrained(str(OUT))

    hist = trainer.state.log_history
    json.dump({"model": MODEL,
               "train": [(h["step"], h["loss"]) for h in hist if "loss" in h],
               "val": [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]},
              open(RESULTS / f"qlora_loss_{TAG}.json", "w"), indent=2)

    shutil.rmtree(CKPT_DIR, ignore_errors=True)      # training done -> free the disk
    print(f"DONE. adapters -> {OUT}")
    print(f"loss history -> results/qlora_loss_{TAG}.json")


if __name__ == "__main__":
    main()