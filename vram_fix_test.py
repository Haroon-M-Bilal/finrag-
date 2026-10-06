"""
Test whether Qwen2.5-7B can train at MAXLEN 3072 on 12 GB with two fixes.

THE PROBLEM, MEASURED

  Qwen2.5-7B   weights 8.44 GB   peak 15.34 GB at 3072   (does not fit)
  Mistral-7B   weights 4.86 GB   peak  8.28 GB at 3072   (fits comfortably)

Same parameter count, 3.6 GB apart in weights alone. The difference is
VOCABULARY SIZE: Qwen has ~152k tokens, Mistral ~33k. Two things scale with it,
and neither is quantized to 4-bit:

  1. The input embedding and output projection matrices. At 152k x 3584 each,
     in float32, that is ~4.4 GB for Qwen against ~1.0 GB for Mistral.
  2. The logits tensor - one score per vocabulary token per position. At
     3072 positions x 152k vocab in float32 that is ~1.9 GB for the forward
     pass alone, and the backward pass needs its gradient too.

FIX 1 - bfloat16 for everything prepare_model_for_kbit_training upcast to
        float32, then float32 restored for the LayerNorms only. Confirmed to
        work on memory: weights 8.44 -> 6.27 GB. The first attempt cast only
        the embedding and output modules and failed with a dtype mismatch -
        the LayerNorms stayed in float32, so float32 hidden states met a
        bfloat16 output layer. Casting everything and then putting the norms
        back keeps dtypes consistent end to end, and norms in float32 are both
        the numerically sensitive part and negligible in size.

FIX 2 - compute the loss in chunks along the sequence instead of letting the
        model materialise logits for all positions at once. The prompt is
        masked out of the loss anyway, so most of those scores are computed
        and discarded.

Each fix is measured separately so we know which one mattered.

Run:  python -u vram_fix_test.py
"""
from __future__ import annotations

import gc

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LIMIT_GB = 10.5          # headroom below the 12 GB card
LENGTHS = [2048, 3072]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)


def build(cast_embeddings: bool):
    m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map={"": 0})
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)

    if cast_embeddings:
        n = 0
        for mod in m.modules():
            w = getattr(mod, "weight", None)
            if w is not None and w.dtype == torch.float32:
                mod.to(torch.bfloat16)
                n += 1
        for name, mod in m.named_modules():
            if "norm" in name.lower():
                mod.to(torch.float32)
        print(f"    cast {n} float32 modules to bfloat16, norms back to fp32")

    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    m.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m.train()
    return m


def chunked_loss(model, ids, labels, chunk=512):
    """
    Loss without materialising logits for the whole sequence at once. Runs the
    transformer body, then applies the output projection and cross-entropy over
    slices of the sequence.
    """
    base = model.base_model.model                    # unwrap PEFT
    hidden = base.model(input_ids=ids).last_hidden_state      # [1, L, H]
    head = base.lm_head

    shift_h = hidden[:, :-1, :]
    shift_l = labels[:, 1:]
    total, n_tok = 0.0, 0
    for s in range(0, shift_h.size(1), chunk):
        h = shift_h[:, s:s + chunk, :]
        l = shift_l[:, s:s + chunk]
        logits = head(h.to(head.weight.dtype)).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               l.reshape(-1), ignore_index=-100,
                               reduction="sum")
        total = total + loss
        n_tok += int((l != -100).sum())
    return total / max(n_tok, 1)


def run(label, cast_embeddings, use_chunked):
    print("=" * 62)
    print(label)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build(cast_embeddings)
    print("    weights: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    for L in LENGTHS:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, 1000, (1, L)).cuda()
            labels = ids.clone()
            labels[:, :L // 2] = -100          # prompt masked, as in training
            if use_chunked:
                loss = chunked_loss(m, ids, labels)
            else:
                loss = m(input_ids=ids, labels=labels).loss
            loss.backward()
            pk = torch.cuda.max_memory_allocated() / 1e9
            print("    %d: peak %.2f GB  %s  (loss %.3f)" %
                  (L, pk, "FITS" if pk < LIMIT_GB else "TOO TIGHT",
                   float(loss)))
            m.zero_grad(set_to_none=True)
        except RuntimeError as e:
            print("    %d: FAIL %s" % (L, str(e)[:70]))
            break

    del m
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run("BASELINE (no fixes)", False, False)
    run("FIX 1 only: bfloat16 non-quantized weights", True, False)
    run("FIX 1 + FIX 2: bfloat16 weights + chunked loss", True, True)
    print("=" * 62)
    print("Mistral-7B for reference: 8.28 GB at 3072")