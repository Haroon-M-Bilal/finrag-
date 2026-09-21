"""
GENERATION EVAL (v6) - two context conditions, numeric exact-match, crash-safe.

DESIGN

  1. TWO CONTEXT CONDITIONS.
       gold      - the evidence chunks. The generation CEILING.
       retrieved - top-3 from ft_dense_ticker_rerank@50, what the deployed
                   pipeline produces (any-gold 0.788).
     The gap is retrieval-attributable error.

  2. NUMERIC EXACT-MATCH on the trusted subset only (gold figure identified by
     a linguistic result-marker AND arithmetic confirmation).

  3. OUTCOMES on the numeric subset: correct / scale error / wrong / abstained
     / malformed / truncated. Truncation is kept separate from malformed so a
     decode-budget artifact is not blamed on the model.

  4. max_new_tokens 768 (measured target p99 645 tokens for Qwen).

  5. LENIENT FOOTER PARSING, leniency reported. Structure only - never scans
     the prose for a number.

  6. CRASH-SAFE: predictions flushed to disk and reloaded on restart. If every
     prediction is already cached, the model is not loaded at all and the run
     is a rescore taking seconds.

HISTORY OF SCORING FIXES

  v5a strip_footer deleted everything from the FIRST "ANSWER" onward. The base
      model tends to open its reply with "ANSWER: ...", so its whole output was
      stripped to an empty string, producing artificially low base scores. It
      now removes only a trailing footer line and a leading "ANSWER:" prefix.

  v5b parse_footer did not recognise the "TEXT" footer used for qualitative
      answers, so every qualitative answer counted as a malformed number. It
      also took the first "ANSWER" match rather than the last.

  v6  ROUNDING-AWARE NUMERIC MATCH. A fixed 0.1% relative tolerance rejected
      correct answers stated at lower precision: gold 2.31%, predicted 2.3% was
      scored wrong. FinDER rounds its own answers, so a prediction is now
      correct if it equals the gold value rounded to the precision the
      prediction itself states. Inspected wrong answers that remain wrong under
      this rule (38.6% predicted as 0.68; 12,120 as 13,676; 6.56% as 0.6%) are
      genuine arithmetic failures, not tolerance artefacts.

Usage:
    python -u src/generate/run_generation_eval.py Qwen/Qwen2.5-3B-Instruct
    python -u src/generate/run_generation_eval.py Qwen/Qwen2.5-3B-Instruct 40
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

PROC = Path("data/finder/processed_v4")
CKPT = Path("checkpoints")
RESULTS = Path("results")
RUNDIR = RESULTS / "runs"
RESULTS.mkdir(exist_ok=True)

BASE = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
TAG = BASE.split("/")[-1].lower().replace(".", "-")
ADAPTER = str(CKPT / f"qlora-{TAG}-v4")
RETRIEVED_RUN = RUNDIR / "ft_dense_ticker_rerank_at_50.trec"

TOPK_CTX = 3
MAX_NEW = 768
BATCH = 4
FLUSH = 40
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SYS = ("You are a financial analyst. Answer the question using only the "
       "provided context from SEC 10-K filings. If the context does not "
       "contain the information needed, say so explicitly. End every "
       "response with a line beginning 'ANSWER:'.")

ANSWER_LINE = re.compile(r"^\s*ANSWER\s*[:\-]\s*(.+?)\s*$",
                         re.I | re.MULTILINE)
STRICT_FOOTER = re.compile(r"^ANSWER: (.+)$", re.MULTILINE)
SCALE_EXP = {"unit": 0, "thousand": 3, "million": 6, "billion": 9,
             "trillion": 12}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def strip_footer(text: str) -> str:
    """Remove a TRAILING footer line and a LEADING 'ANSWER:' prefix."""
    t = (text or "").strip()
    lines = t.split("\n")
    if len(lines) > 1 and re.match(r"\s*ANSWER\s*[:\-]", lines[-1], re.I):
        t = "\n".join(lines[:-1]).strip()
    return re.sub(r"^\s*ANSWER\s*[:\-]\s*", "", t, flags=re.I).strip()


def parse_footer(text: str):
    """
    -> ("value", mantissa, scale_exp, unit) | ("na",) | ("insufficient",)
       | ("malformed",)
    Uses the LAST 'ANSWER:' line. Structure only - never scans the prose.
    """
    hits = ANSWER_LINE.findall(text or "")
    if not hits:
        return ("malformed",)
    payload = hits[-1].strip()
    up = payload.upper()
    if up.startswith(("N/A", "NA", "TEXT")):
        return ("na",)
    if up.startswith("INSUFFICIENT"):
        return ("insufficient",)
    parts = [p.strip() for p in re.split(r"[|,\t]", payload) if p.strip()]
    try:
        mant = float(parts[0].replace("$", "").replace(",", ""))
    except (ValueError, IndexError):
        return ("malformed",)
    scale = SCALE_EXP.get(parts[1].lower(), 0) if len(parts) > 1 else 0
    unit = parts[2].upper() if len(parts) > 2 else "USD"
    return ("value", mant, scale, unit)


def _decimals(x: float) -> int:
    """Decimal places the prediction states. Scientific notation -> 0."""
    s = repr(float(x))
    if "e" in s.lower():
        return 0
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def numeric_outcome(pred: str, gold_target: dict, truncated: bool):
    """
    Correct if the prediction equals the gold value rounded to the precision the
    prediction itself states. FinDER rounds its own answers, so 2.3% for a true
    2.31% is correct at the precision chosen.
    """
    if truncated:
        return "truncated"
    p = parse_footer(pred)
    if p[0] == "malformed":
        return "malformed"
    if p[0] in ("na", "insufficient"):
        return "abstained"
    _, mant, scale, _ = p
    d = _decimals(mant)
    half = 0.5 * 10 ** (-d)
    gold_in_pred_scale = gold_target["value"] / (10 ** scale)
    if abs(round(gold_in_pred_scale, d) - round(mant, d)) < half:
        return "correct"
    if abs(round(gold_target["mantissa"], d) - round(mant, d)) < half:
        return "scale_error"
    return "wrong"


def compute_metrics(preds, golds):
    from rouge_score import rouge_scorer
    import sacrebleu
    from bert_score import score as bertscore
    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                  use_stemmer=True)
    r1, r2, rl = [], [], []
    for g, p in zip(golds, preds):
        s = rs.score(g, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    bleu = float(np.mean([sacrebleu.sentence_bleu(p or " ", [g]).score
                          for g, p in zip(golds, preds)])) / 100.0
    safe = [p if p.strip() else "." for p in preds]
    _, _, F = bertscore(safe, golds, lang="en", verbose=False)
    return {"ROUGE-1": float(np.mean(r1)), "ROUGE-2": float(np.mean(r2)),
            "ROUGE-L": float(np.mean(rl)), "BLEU": bleu,
            "BERTScore-F1": float(F.mean())}


def load_retrieved(path, k):
    run = {}
    if not path.exists():
        print(f"  WARNING: {path} missing - retrieved condition skipped")
        return run
    for line in open(path, encoding="utf-8"):
        qid, _, cid, rank, _, _ = line.split()
        run.setdefault(qid, []).append((int(rank), cid))
    return {q: [c for _, c in sorted(v)][:k] for q, v in run.items()}


def main():
    if not Path(ADAPTER).exists():
        raise SystemExit(f"adapter not found: {ADAPTER}")
    print(f"MODEL: {BASE}\nADAPTER: {ADAPTER}\ndevice: {DEV}")

    corpus = {r["_id"]: r["text"] for r in jl(PROC / "corpus.jsonl")}
    numeric = {r["_id"]: r for r in jl(PROC / "answers_numeric.jsonl")}
    rows = jl(PROC / "qa_sft_eval.jsonl")
    if LIMIT:
        rows = rows[:LIMIT]
        print(f"LIMIT: {len(rows)} questions")

    retrieved = load_retrieved(RETRIEVED_RUN, TOPK_CTX)
    trusted = {q for q, r in numeric.items()
               if r.get("tier") in ("anchored_verified", "single_figure")}
    n_trusted = sum(1 for r in rows if r["qid"] in trusted)
    print(f"eval questions: {len(rows)}   numeric (trusted): {n_trusted}")

    cache_path = RESULTS / f"preds_{TAG}_v4.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}

    keys = []
    for cond in ("gold", "retrieved"):
        if cond == "retrieved" and not retrieved:
            continue
        for ft in (True, False):
            keys.append((cond, ft, f"{cond}_{'ft' if ft else 'base'}"))

    need_model = any(any(r["qid"] not in cache.get(k, {}) for r in rows)
                     for _, _, k in keys)

    model = tok = None
    if need_model:
        tok = AutoTokenizer.from_pretrained(BASE)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        print("loading base model (4-bit)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=bnb, device_map={"": 0})
        model.config.use_cache = True
        model = PeftModel.from_pretrained(model, ADAPTER)
        model.eval()
    else:
        print("all predictions cached - rescoring only, no model loaded")

    def prompt_for(row, condition):
        if condition == "gold":
            chunks = row["chunk_texts"]
        else:
            cids = retrieved.get(row["qid"], [])
            chunks = [corpus[c] for c in cids if c in corpus] or \
                     ["(no context retrieved)"]
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
        msgs = [{"role": "system", "content": SYS},
                {"role": "user",
                 "content": f"Context:\n{ctx}\n\nQuestion: {row['question']}"}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)

    def generate(condition, use_ft, key):
        done = cache.get(key, {})
        todo = [r for r in rows if r["qid"] not in done]
        if not todo:
            print(f"  {key}: cached")
            return done
        print(f"  {key}: {len(todo)} to generate")
        for s in range(0, len(todo), BATCH):
            batch = todo[s:s + BATCH]
            texts = [prompt_for(r, condition) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=3072).to(model.device)
            with torch.no_grad():
                if use_ft:
                    out = model.generate(**enc, max_new_tokens=MAX_NEW,
                                         do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                else:
                    with model.disable_adapter():
                        out = model.generate(**enc, max_new_tokens=MAX_NEW,
                                             do_sample=False,
                                             pad_token_id=tok.pad_token_id)
            for i, r in enumerate(batch):
                gen = out[i][enc["input_ids"].shape[1]:]
                n_new = int((gen != tok.pad_token_id).sum())
                done[r["qid"]] = {
                    "text": tok.decode(gen, skip_special_tokens=True).strip(),
                    "truncated": n_new >= MAX_NEW - 1}
            if (s // BATCH) % FLUSH == 0:
                cache[key] = done
                json.dump(cache, open(cache_path, "w"))
                print(f"    {s + len(batch)}/{len(todo)}", end="\r")
        print()
        cache[key] = done
        json.dump(cache, open(cache_path, "w"))
        return done

    conditions = {k: generate(c, ft, k) for c, ft, k in keys}

    if model is not None:
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- scoring
    golds_full = {r["qid"]: r["target"] for r in rows}
    report = {}
    for key, preds in conditions.items():
        ids = [r["qid"] for r in rows if r["qid"] in preds]
        gold_txt = [strip_footer(golds_full[q]) for q in ids]
        pred_txt = [strip_footer(preds[q]["text"]) for q in ids]
        m = compute_metrics(pred_txt, gold_txt)

        strict = sum(1 for q in ids if STRICT_FOOTER.search(preds[q]["text"]))
        parsed = sum(1 for q in ids
                     if parse_footer(preds[q]["text"])[0] != "malformed")
        trunc = sum(1 for q in ids if preds[q]["truncated"])
        abstain_all = sum(1 for q in ids
                          if parse_footer(preds[q]["text"])[0]
                          == "insufficient")

        outcomes = {"correct": 0, "scale_error": 0, "wrong": 0,
                    "abstained": 0, "malformed": 0, "truncated": 0}
        for q in ids:
            if q not in trusted:
                continue
            tgt = numeric[q].get("target")
            if tgt:
                outcomes[numeric_outcome(preds[q]["text"], tgt,
                                         preds[q]["truncated"])] += 1
        n_num = sum(outcomes.values())

        report[key] = {
            **m, "n": len(ids),
            "strict_format_rate": strict / max(len(ids), 1),
            "parsed_rate": parsed / max(len(ids), 1),
            "truncated_rate": trunc / max(len(ids), 1),
            "abstain_rate": abstain_all / max(len(ids), 1),
            "numeric_n": n_num, "numeric_outcomes": outcomes,
            "numeric_em": outcomes["correct"] / max(n_num, 1),
            "numeric_em_scale_lenient":
                (outcomes["correct"] + outcomes["scale_error"])
                / max(n_num, 1),
        }

    # -------------------------------------------------------------- output
    cols = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore-F1"]
    L = [f"# Generation eval: {BASE}", "",
         f"{len(rows)} held-out test questions from filings never seen in "
         f"training. `gold` = evidence chunks (ceiling); `retrieved` = "
         f"top-{TOPK_CTX} from the deployed pipeline. Footer lines are "
         f"stripped before ROUGE/BERTScore.", "",
         "| condition | " + " | ".join(cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for k in sorted(report):
        v = report[k]
        L.append(f"| {k} | " + " | ".join(f"{v[c]:.4f}" for c in cols) + " |")

    L += ["", "## Structured output", "",
          "| condition | strict fmt | parsed | abstain | truncated |",
          "|---|---|---|---|---|"]
    for k in sorted(report):
        v = report[k]
        L.append(f"| {k} | {v['strict_format_rate']:.3f} | "
                 f"{v['parsed_rate']:.3f} | {v['abstain_rate']:.3f} | "
                 f"{v['truncated_rate']:.3f} |")

    L += ["", "## Numeric exact-match (trusted subset, rounding-aware)", "",
          "A prediction is correct if it equals the gold value rounded to the "
          "precision the prediction states.", "",
          "| condition | n | EM | EM scale-lenient | correct | scale err | "
          "wrong | abstained | malformed | truncated |",
          "|" + "---|" * 10]
    for k in sorted(report):
        v = report[k]
        o = v["numeric_outcomes"]
        L.append(f"| {k} | {v['numeric_n']} | {v['numeric_em']:.3f} | "
                 f"{v['numeric_em_scale_lenient']:.3f} | {o['correct']} | "
                 f"{o['scale_error']} | {o['wrong']} | {o['abstained']} | "
                 f"{o['malformed']} | {o['truncated']} |")

    out = "\n".join(L) + "\n"
    print("\n" + out)
    (RESULTS / f"generation_{TAG}_v4.md").write_text(out, encoding="utf-8")
    json.dump({"model": BASE, "max_new_tokens": MAX_NEW, "topk_ctx": TOPK_CTX,
               "n_questions": len(rows), "numeric_rule": "rounding-aware",
               "report": report},
              open(RESULTS / f"generation_{TAG}_v4.json", "w"), indent=2)
    print(f"saved -> results/generation_{TAG}_v4.md / .json")


if __name__ == "__main__":
    main()