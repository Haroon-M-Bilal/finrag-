"""
GENERATOR fine-tuning (QLoRA) - multi-model, checkpointed, auto-resuming.

Usage:
    python -u src/generate/finetune_qlora.py Qwen/Qwen2.5-3B-Instruct
    python -u src/generate/finetune_qlora.py Qwen/Qwen2.5-7B-Instruct
    python -u src/generate/finetune_qlora.py mistralai/Mistral-7B-Instruct-v0.3

DESIGN NOTES

  MAXLEN 3072. Measured on the rebuilt training data, prompt+target averages
  1,915 tokens for Qwen and 2,092 for Mistral. The previous setting of 1024 fit
  ZERO examples, so every target was truncated. 3072 was measured at 8.78 GB
  peak for Qwen-3B and 8.28 GB for Mistral-7B on a 12 GB card.

  TRUNCATION NEVER TOUCHES THE TARGET. The target is reserved first and
  context is trimmed around it by dropping whole chunks, never mid-sentence.

  VALIDATION IS A SEEDED RANDOM SAMPLE, not the last 200 rows (which followed
  filing order and so came from a handful of filings).

RESUME CORRECTNESS

  save_only_model is now FALSE, so checkpoints carry optimizer and learning-
  rate scheduler state and a resumed run continues exactly. It had been set to
  True on the guess that the large optimizer write triggered the machine's
  blue screens. BlueScreenView later attributed 4 of 5 crashes to nvlddmkm.sys
  (the NVIDIA display driver), fixed by a clean driver reinstall - so the
  workaround was unnecessary, and it had a cost:

  Resuming from a model-only checkpoint RESTARTS THE LEARNING-RATE SCHEDULE
  FROM STEP 0. Measured on this project: the Qwen-3B resume from checkpoint-400
  logged a learning rate of 8.33e-05 at step 410 - identical to step 10 of the
  original run - i.e. warmup restarted and the final steps ran near peak LR.

  For checkpoints already written without scheduler state, the schedule is now
  FAST-FORWARDED to the checkpoint's step before training continues, and the
  resulting learning rate is printed so it can be checked against the original
  log. Adam's moment estimates still restart (they were not saved), which costs
  a small transient bump but does not distort the schedule.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import warnings
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

PROC = Path("data/finder/processed_v4")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
TAG = MODEL.split("/")[-1].lower().replace(".", "-")
OUT = Path(f"checkpoints/qlora-{TAG}-v4")
CKPT_DIR = Path(f"checkpoints/_tmp_{TAG}_v4")

MAXLEN = 3072
EPOCHS, GRAD_ACCUM, LR = 2, 16, 2e-4
SAVE_EVERY = 100
EVAL_EVERY = 50
VAL_N = 200
SEED = 42

SYS = ("You are a financial analyst. Answer the question using only the "
       "provided context from SEC 10-K filings. If the context does not "
       "contain the information needed, say so explicitly. End every "
       "response with a line beginning 'ANSWER:'.")

_dropped = {"rows": 0, "chunks": 0, "hard_trunc": 0}


def build(tok, ex):
    """Reserve the target, then fit context around it by dropping whole chunks."""
    target = ex["target"] + tok.eos_token
    ans_ids = tok(target, add_special_tokens=False)["input_ids"]

    chunks = list(ex["chunk_texts"])
    dropped_here = 0
    while True:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
        msgs = [{"role": "system", "content": SYS},
                {"role": "user",
                 "content": f"Context:\n{ctx}\n\nQuestion: {ex['question']}"}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=True)
        if len(prompt_ids) + len(ans_ids) <= MAXLEN:
            break
        if len(chunks) > 1:
            chunks.pop()
            dropped_here += 1
            continue
        budget = MAXLEN - len(ans_ids) - 64
        cid = tok(chunks[0], add_special_tokens=False)["input_ids"][:max(budget, 32)]
        chunks[0] = tok.decode(cid, skip_special_tokens=True)
        _dropped["hard_trunc"] += 1
        msgs = [{"role": "system", "content": SYS},
                {"role": "user",
                 "content": f"Context:\n[1] {chunks[0]}\n\n"
                            f"Question: {ex['question']}"}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=True)
        break

    if dropped_here:
        _dropped["rows"] += 1
        _dropped["chunks"] += dropped_here

    ids = prompt_ids + ans_ids
    labels = [-100] * len(prompt_ids) + ans_ids
    return {"input_ids": ids[:MAXLEN], "labels": labels[:MAXLEN]}


class Collator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ii, ll, am = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ii.append(b["input_ids"] + [self.pad] * n)
            ll.append(b["labels"] + [-100] * n)
            am.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}


class ResumeTrainer(Trainer):
    """
    If resuming from a checkpoint that holds no scheduler state, advance the
    freshly-created LR schedule to the checkpoint's step, so training continues
    where it left off instead of restarting warmup.
    """
    resume_step = 0

    def create_scheduler(self, num_training_steps, optimizer=None):
        sched = super().create_scheduler(num_training_steps, optimizer)
        if self.resume_step:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # step-before-optimizer notice
                for _ in range(self.resume_step):
                    sched.step()
            print(f"  LR schedule fast-forwarded {self.resume_step} steps "
                  f"-> lr {sched.get_last_lr()[0]:.4e}")
            self.resume_step = 0
        return sched


def latest_checkpoint(d: Path):
    """Newest COMPLETE checkpoint. A crash mid-save leaves a partial folder."""
    if not d.exists():
        return None
    cks = sorted((p for p in d.glob("checkpoint-*") if p.is_dir()),
                 key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for c in cks:
        if (c / "trainer_state.json").exists():
            return c
        print(f"  skipping incomplete checkpoint {c.name} "
              f"(no trainer_state.json)")
    return None


def main():
    print(f"MODEL: {MODEL} -> {OUT}   (maxlen={MAXLEN})")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map={"": 0})
    model = prepare_model_for_kbit_training(model,
                                            use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in
            open(PROC / "qa_sft_train.jsonl", encoding="utf-8")]
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    val_idx = set(idx[:VAL_N])
    val_rows = [rows[i] for i in sorted(val_idx)]
    train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]

    train_data = [build(tok, r) for r in train_rows]
    val_data = [build(tok, r) for r in val_rows]
    lens = [len(d["input_ids"]) for d in train_data]
    print(f"train: {len(train_data)}  val: {len(val_data)}")
    print(f"seq len: mean {sum(lens) / len(lens):.0f}  max {max(lens)}")
    print(f"context trimmed on {_dropped['rows']} rows "
          f"({_dropped['chunks']} chunks dropped, "
          f"{_dropped['hard_trunc']} hard truncations)")

    args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS, learning_rate=LR, bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", warmup_ratio=0.05, report_to="none",
        logging_steps=10,
        eval_strategy="steps", eval_steps=EVAL_EVERY,
        per_device_eval_batch_size=1,
        save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=2,
        save_safetensors=True,
        save_only_model=False,     # keep optimizer + scheduler: exact resume
        seed=SEED,
    )
    trainer = ResumeTrainer(model=model, args=args, train_dataset=train_data,
                            eval_dataset=val_data, data_collator=Collator(tok))

    resume = latest_checkpoint(CKPT_DIR)
    if resume:
        step = int(resume.name.split("-")[-1])
        if not (resume / "scheduler.pt").exists():
            trainer.resume_step = step
            print(f"RESUMING from {resume} (no scheduler state saved - "
                  f"LR will be fast-forwarded to step {step})")
        else:
            print(f"RESUMING from {resume} (full state)")
        trainer.train(resume_from_checkpoint=str(resume))
    else:
        print("starting fresh")
        trainer.train()

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))

    hist = trainer.state.log_history
    json.dump({"model": MODEL, "maxlen": MAXLEN, "epochs": EPOCHS,
               "grad_accum": GRAD_ACCUM, "lr": LR, "seed": SEED,
               "source": str(PROC),
               "seq_len_mean": sum(lens) / len(lens), "seq_len_max": max(lens),
               "context_trimmed_rows": _dropped["rows"],
               "train": [(h["step"], h["loss"]) for h in hist if "loss" in h],
               "val": [(h["step"], h["eval_loss"]) for h in hist
                       if "eval_loss" in h]},
              open(RESULTS / f"qlora_loss_{TAG}_v4.json", "w"), indent=2)

    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    print(f"DONE. adapters -> {OUT}")


if __name__ == "__main__":
    main()