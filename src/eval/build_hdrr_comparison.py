"""
Generate FinRAG answers for the 259 queries we share with Cheng et al. (HDRR).

NO API COST. Local GPU only. This produces the answers; judging is a separate
script that needs OpenAI credit.

WHY NO RETRIEVAL RUNS HERE

results/runs/ft_dense_ticker_rerank_at_50.trec already holds the top-100 for
every test question under the winning configuration (NDCG@10 0.5414). The 259
shared queries are a subset of those 1,088, so retrieval is a file read. This
also guarantees the answers come from exactly the pipeline that produced the
published retrieval numbers - if this script re-ran retrieval it could drift and
the two halves of the paper would stop agreeing.

WHAT IS MATCHED TO THEIR HARNESS, AND WHY

  1. GROUND TRUTH IS READ FROM THEIR OWN BENCHMARK CSVs, not re-derived from
     HuggingFace. Their judge scored against the `ground_truth` column in those
     files, so taking it verbatim removes any chance of grading our answers
     against a slightly different gold string than theirs were graded against.
     Consistency across their systems is checked and any disagreement reported.

  2. THEIR TEXT SANITISER IS REPLICATED. main_benchmark_rag_result.py rewrites
     newlines as literal backslash-n and folds smart quotes, dashes and other
     Unicode to ASCII BEFORE the row is written, and the judge reads that CSV.
     So their judge never saw raw model output - it saw sanitised text. Ours is
     sanitised identically.

  3. OUTPUT USES THEIR INPUT COLUMN NAMES
     (original_row, query_id, query, tickers, file_paths, missing_files,
     final_answer, error) so their unmodified benchmark script can consume this
     file if a reviewer asks us to run their code rather than our own.

  4. FOOTERS ARE STRIPPED with the same strip_footer as the generation eval.
     Our models were trained to end with an 'ANSWER:' line; that is our
     structured-output channel, not prose, and their judge would read it as
     noise.

THE ABSTENTION PROBLEM - READ THIS BEFORE SPENDING CREDIT

Their rubric (BENCHMARK_EVALUATION_SYSTEM_PROMPT) says:

    "If the answer says it cannot find the information but the ground truth
     exists, score should be low (1-3)"

Our generators were deliberately calibrated to abstain (ABSTAIN_FRAC = 0.06 in
prepare_qa_data v5). Under their rubric an honest abstention is scored as
harshly as a confident fabrication. Every abstention therefore costs us roughly
5-7 points against a correct answer, for a reason that has nothing to do with
retrieval quality.

This script reports the abstention rate on the 259 BEFORE any money is spent,
counted two ways: structurally (the ANSWER footer says INSUFFICIENT) and in the
prose (refusal phrasing). If that rate is high, we decide how to report it
rather than discovering it in the final table.

Usage:
    python -u src/eval/build_hdrr_comparison.py
    python -u src/eval/build_hdrr_comparison.py Qwen/Qwen2.5-3B-Instruct
    python -u src/eval/build_hdrr_comparison.py mistralai/Mistral-7B-Instruct-v0.3 gold

The optional second argument picks the context condition:
    retrieved (default) - top-3 from the deployed pipeline. THE SYSTEM.
    gold                - the evidence chunks. A ceiling, not a system; use it
                          only to separate retrieval error from generation
                          error, and label it as a ceiling in the paper.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

PROC = Path("data/finder/processed_v4")
CKPT = Path("checkpoints")
RESULTS = Path("results")
RUNDIR = RESULTS / "runs"
ODIR = Path("external/hdrr/output")
USABLE = RESULTS / "hdrr_usable_queries.txt"


BASE = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
CONDITION = sys.argv[2] if len(sys.argv) > 2 else "retrieved"
TAG = BASE.split("/")[-1].lower().replace(".", "-")
ADAPTER = str(CKPT / f"qlora-{TAG}-v4")

# Context depth, and the input budget that must grow with it.
#
# TOPK_CTX=3 at MAXLEN=3072 was the deployed setting and is the default, so
# re-running with no arguments reproduces the published numbers exactly.
#
# Raising TOPK_CTX WITHOUT raising MAXLEN silently truncates: chunks are ~448
# BGE tokens, which run longer under Mistral's tokenizer, so past about six
# chunks the tail of the context is cut off and the extra retrieval depth buys
# nothing. The two move together.
#
# VRAM: the KV cache grows linearly in sequence length and in batch size. On
# 12.9 GB with a 4-bit 7B, TOPK_CTX=10 / MAXLEN=6144 wants BATCH=2. If it OOMs,
# drop BATCH to 1 before touching anything else.
#
# CAUTION, and this must be measured rather than assumed: the adapter was
# fine-tuned on roughly two gold chunks per example. Ten chunks, most of them
# distractors, is a train/test mismatch that may LOWER scores on queries the
# model currently answers well, even as it rescues queries it currently
# misses. Compare per-query scores against the depth-3 run before believing
# any improvement in the mean.
TOPK_CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAXLEN = int(sys.argv[4]) if len(sys.argv) > 4 else 3072
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 4

# Decode budget. Longer contexts make the model more discursive: at depth 3
# only 2/258 answers hit the cap, at depth 10 it was 10/258. A truncated answer
# is scored as a bad answer by the judge, so the cap must grow with the context
# or deeper retrieval is partly self-defeating.
MAX_NEW = int(sys.argv[6]) if len(sys.argv) > 6 else 768

# Which retrieval run feeds the generator. Defaults to the original regex-routed
# run so earlier results stay reproducible; pass the v2 file to use SEC-name
# routing (NDCG@10 0.5414 -> 0.5994). The run name is carried into every output
# filename so the two can never be confused for one another.
RUN_NAME = sys.argv[7] if len(sys.argv) > 7 else "ft_dense_ticker_rerank_at_50"
FLUSH = 10
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# depth-3 keeps the original filenames so nothing already produced is clobbered
RETRIEVED_RUN = RUNDIR / f"{RUN_NAME}.trec"

SUFFIX = ("" if TOPK_CTX == 3 else f"_k{TOPK_CTX}") + \
         ("_v2" if RUN_NAME.endswith("_v2") else "")

# identical to run_generation_eval.py - the model was trained against this
SYS = ("You are a financial analyst. Answer the question using only the "
       "provided context from SEC 10-K filings. If the context does not "
       "contain the information needed, say so explicitly. End every "
       "response with a line beginning 'ANSWER:'.")

ANSWER_LINE = re.compile(r"^\s*ANSWER\s*[:\-]\s*(.+?)\s*$", re.I | re.MULTILINE)
REFUSAL_RE = re.compile(
    r"cannot find|can't find|not find sufficient|insufficient information|"
    r"does not contain|doesn't contain|not available in the (provided|given)|"
    r"no information (is )?(provided|available)|unable to (determine|answer)",
    re.I)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# their sanitiser, verbatim from main_benchmark_rag_result.py
UNICODE_REPLACEMENTS = {
    '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
    '\u2013': '-', '\u2014': '--', '\u00D7': 'x', '\u00F7': '/',
    '\u2212': '-', '\u00B1': '+/-', '\u2026': '...', '\u2022': '-',
    '\u00B0': ' deg', '\u00A0': ' ',
}


def sanitize_text(value) -> str:
    """Byte-for-byte reimplementation of their StreamingCSVWriter._sanitize_text.

    Their judge read the CSV this produced, so this is the text their grader
    actually saw. Ours must pass through the same transformation or we are
    comparing differently-shaped inputs.
    """
    if value is None:
        return ''
    s = str(value)
    s = s.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
    for uni, ascii_eq in UNICODE_REPLACEMENTS.items():
        s = s.replace(uni, ascii_eq)
    return s


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def strip_footer(text: str) -> str:
    """Remove a TRAILING footer line and a LEADING 'ANSWER:' prefix.

    Same as run_generation_eval.py v7. Do not 'simplify' this: an earlier
    version cut from the FIRST 'ANSWER' onward, which emptied outputs that
    opened with 'ANSWER:' and silently produced bogus scores.
    """
    t = (text or "").strip()
    lines = t.split("\n")
    if len(lines) > 1 and re.match(r"\s*ANSWER\s*[:\-]", lines[-1], re.I):
        t = "\n".join(lines[:-1]).strip()
    return re.sub(r"^\s*ANSWER\s*[:\-]\s*", "", t, flags=re.I).strip()


def footer_kind(text: str) -> str:
    """'insufficient' | 'na' | 'value' | 'malformed' - structure only."""
    hits = ANSWER_LINE.findall(text or "")
    if not hits:
        return "malformed"
    up = hits[-1].strip().upper()
    if up.startswith("INSUFFICIENT"):
        return "insufficient"
    if up.startswith(("N/A", "NA", "TEXT")):
        return "na"
    return "value"


def load_retrieved(path, k):
    if not path.exists():
        raise SystemExit(
            f"missing: {path}\nRun src/eval/run_ablation_split.py first - the "
            f"comparison must use the same run file as the published "
            f"retrieval numbers.")
    run = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        qid, _, cid, rank, _, _ = line.split()
        run[qid].append((int(rank), cid))
    return {q: [c for _, c in sorted(v)][:k] for q, v in run.items()}


def load_their_data(usable: set[str]):
    """
    qid -> {"ground_truth", "hybrid_answer", "hybrid_score"}

    Ground truth is taken from their files. Preference order: a hybrid run
    (our comparison target), then any other system. Disagreement between
    systems on the same qid is reported, never silently resolved.
    """
    gt_by_qid: dict[str, dict[str, str]] = defaultdict(dict)
    hybrid: dict[str, dict] = {}

    files = sorted(ODIR.glob("benchmark-*.csv"))
    if not files:
        raise SystemExit(f"no benchmark CSVs in {ODIR}")

    for p in files:
        is_hybrid = "-hybrid-" in p.name
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                qid = (r.get("query_id") or "").strip()
                if qid not in usable:
                    continue
                gt = (r.get("ground_truth") or "").strip()
                if gt:
                    gt_by_qid[qid][gt] = p.name
                if is_hybrid:
                    sc = (r.get("score") or "").strip()
                    hybrid.setdefault(qid, {
                        "answer": (r.get("rag_answer") or "").strip(),
                        "score": float(sc) if sc else None,
                        "query": (r.get("query") or "").strip(),
                    })

    conflicts = {q: list(v) for q, v in gt_by_qid.items() if len(v) > 1}
    out = {}
    for qid, variants in gt_by_qid.items():
        # longest variant: sanitisation only ever shortens or substitutes
        gt = max(variants, key=len)
        out[qid] = {
            "ground_truth": gt,
            "hybrid_answer": hybrid.get(qid, {}).get("answer", ""),
            "hybrid_score": hybrid.get(qid, {}).get("score"),
            "their_query": hybrid.get(qid, {}).get("query", ""),
        }
    return out, conflicts


def main():
    if CONDITION not in ("retrieved", "gold"):
        raise SystemExit("second argument must be 'retrieved' or 'gold'")
    if not Path(ADAPTER).exists():
        raise SystemExit(f"adapter not found: {ADAPTER}")
    if not USABLE.exists():
        raise SystemExit(f"missing {USABLE} - run check_hdrr_overlap.py first")

    usable = [l.strip() for l in open(USABLE) if l.strip()]
    usable_set = set(usable)
    print(f"MODEL: {BASE}\nADAPTER: {ADAPTER}\nCONDITION: {CONDITION}\n"
          f"TOPK_CTX: {TOPK_CTX}   MAXLEN: {MAXLEN}   BATCH: {BATCH}   "
          f"MAX_NEW: {MAX_NEW}\n"
          f"RUN: {RETRIEVED_RUN.name}\n"
          f"device: {DEV}\nshared queries: {len(usable)}")

    # ------------------------------------------------------- our side of it
    corpus = {r["_id"]: r["text"] for r in jl(PROC / "corpus.jsonl")}
    qtext = {str(r["_id"]): r["text"]
             for r in jl(PROC / "queries_test.jsonl")
             if str(r["_id"]) in usable_set}

    retrieved = load_retrieved(RETRIEVED_RUN, TOPK_CTX)

    gold_chunks = {}
    if CONDITION == "gold":
        with open(PROC / "qrels_test.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                a, b, _ = line.rstrip("\n").split("\t")
                if a in usable_set:
                    gold_chunks.setdefault(a, []).append(b)

    missing_q = usable_set - set(qtext)
    missing_r = usable_set - set(retrieved)
    if missing_q:
        print(f"  WARNING: {len(missing_q)} shared queries have no text in "
              f"queries_test.jsonl - dropped")
    if missing_r and CONDITION == "retrieved":
        print(f"  WARNING: {len(missing_r)} shared queries absent from the run "
              f"file - dropped")

    # ----------------------------------------------------- their side of it
    theirs, conflicts = load_their_data(usable_set)
    print(f"ground truth recovered for {len(theirs)}/{len(usable)} queries")
    if conflicts:
        print(f"  WARNING: {len(conflicts)} queries have differing ground "
              f"truth across their systems; using the longest variant")
    no_hybrid = [q for q in theirs if theirs[q]["hybrid_score"] is None]
    if no_hybrid:
        print(f"  NOTE: {len(no_hybrid)} queries have no hybrid score - "
              f"excluded from the paired test later")

    ids = [q for q in usable
           if q in qtext and q in theirs
           and (q in retrieved if CONDITION == "retrieved"
                else q in gold_chunks)]
    print(f"\nqueries to answer: {len(ids)}")
    if not ids:
        raise SystemExit("nothing to do")

    # ------------------------------------------------------------- generate
    cache_path = RESULTS / f"hdrr_preds_{TAG}_{CONDITION}{SUFFIX}.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    todo = [q for q in ids if q not in cache]
    print(f"cached: {len(ids) - len(todo)}   to generate: {len(todo)}")

    if todo:
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

        def prompt_for(qid):
            if CONDITION == "gold":
                chunks = [corpus[c] for c in gold_chunks[qid] if c in corpus]
            else:
                chunks = [corpus[c] for c in retrieved[qid] if c in corpus]
            chunks = chunks or ["(no context retrieved)"]
            ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user",
                     "content": f"Context:\n{ctx}\n\nQuestion: {qtext[qid]}"}]
            return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False)

        for s in range(0, len(todo), BATCH):
            batch = todo[s:s + BATCH]
            enc = tok([prompt_for(q) for q in batch], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=MAXLEN).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for i, q in enumerate(batch):
                gen = out[i][enc["input_ids"].shape[1]:]
                n_new = int((gen != tok.pad_token_id).sum())
                cache[q] = {
                    "text": tok.decode(gen, skip_special_tokens=True).strip(),
                    "truncated": n_new >= MAX_NEW - 1}
            if (s // BATCH) % FLUSH == 0:
                json.dump(cache, open(cache_path, "w"))
                print(f"  {s + len(batch)}/{len(todo)}", end="\r")
        print()
        json.dump(cache, open(cache_path, "w"))
        del model
        torch.cuda.empty_cache()
    else:
        print("all answers cached - no model loaded")

    # -------------------------------------------------- assemble + diagnose
    rows, bundle = [], {}
    n_abstain_footer = n_abstain_prose = n_trunc = n_empty = 0

    for i, qid in enumerate(ids, 1):
        raw = cache[qid]["text"]
        prose = strip_footer(raw)
        kind = footer_kind(raw)

        abst_f = kind == "insufficient"
        abst_p = bool(REFUSAL_RE.search(prose))
        n_abstain_footer += abst_f
        n_abstain_prose += abst_p
        n_trunc += bool(cache[qid]["truncated"])
        n_empty += (not prose.strip())

        rows.append({
            "original_row": i,
            "query_id": qid,
            "query": sanitize_text(qtext[qid]),
            "tickers": qid.split("::")[0] if "::" in qid else "",
            "file_paths": "|".join(retrieved.get(qid, [])[:TOPK_CTX]),
            "missing_files": "",
            "final_answer": sanitize_text(prose),
            "error": "",
        })
        bundle[qid] = {
            "question": qtext[qid],
            "ground_truth": theirs[qid]["ground_truth"],
            "our_answer": sanitize_text(prose),
            "our_raw": raw,
            "footer_kind": kind,
            "abstained_footer": abst_f,
            "abstained_prose": abst_p,
            "truncated": bool(cache[qid]["truncated"]),
            "context_cids": (gold_chunks.get(qid, []) if CONDITION == "gold"
                             else retrieved.get(qid, [])[:TOPK_CTX]),
            "their_hybrid_answer": theirs[qid]["hybrid_answer"],
            "their_hybrid_score": theirs[qid]["hybrid_score"],
        }

    csv_path = RESULTS / f"finrag_rag_results_{TAG}_{CONDITION}{SUFFIX}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    bundle_path = RESULTS / f"hdrr_comparison_{TAG}_{CONDITION}{SUFFIX}.json"
    json.dump({
        "model": BASE, "adapter": ADAPTER, "condition": CONDITION,
        "topk_ctx": TOPK_CTX, "maxlen": MAXLEN, "batch": BATCH,
        "max_new_tokens": MAX_NEW,
        "n": len(ids),
        "run_file": str(RETRIEVED_RUN), "run_name": RUN_NAME,
        "ground_truth_source": "external/hdrr/output benchmark CSVs, verbatim",
        "sanitiser": "replicates their StreamingCSVWriter._sanitize_text",
        "ground_truth_conflicts": conflicts,
        "queries": bundle,
    }, open(bundle_path, "w"), indent=2)

    n = len(ids)
    hyb = [v["their_hybrid_score"] for v in bundle.values()
           if v["their_hybrid_score"] is not None]

    print(f"\n{'=' * 64}")
    print(f"ANSWERS BUILT  ({n} queries, condition={CONDITION})")
    print(f"{'=' * 64}")
    print(f"  abstained (ANSWER footer says INSUFFICIENT): "
          f"{n_abstain_footer:>4}  {100 * n_abstain_footer / n:5.1f}%")
    print(f"  refusal phrasing in the prose              : "
          f"{n_abstain_prose:>4}  {100 * n_abstain_prose / n:5.1f}%")
    print(f"  truncated at {MAX_NEW} new tokens            : "
          f"{n_trunc:>4}  {100 * n_trunc / n:5.1f}%")
    print(f"  empty after footer strip                   : "
          f"{n_empty:>4}  {100 * n_empty / n:5.1f}%")
    if hyb:
        print(f"\n  their hybrid on these queries: mean "
              f"{sum(hyb) / len(hyb):.2f}  (n={len(hyb)})")

    worst = 100 * max(n_abstain_footer, n_abstain_prose) / n
    print(f"\n{'-' * 64}")
    if worst >= 20:
        print(f"ABSTENTION {worst:.1f}% - their rubric scores every one of "
              f"these 1-3.\nExpect a large penalty unattributable to "
              f"retrieval. Decide how to\nreport it BEFORE judging.")
    elif worst >= 8:
        print(f"ABSTENTION {worst:.1f}% - material but survivable. Report it "
              f"alongside the\nmean so the penalty is visible.")
    else:
        print(f"ABSTENTION {worst:.1f}% - low enough that the rubric's "
              f"abstention clause\nwill not drive the result.")
    print(f"{'-' * 64}")

    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {bundle_path}")
    print(f"\nNow read 15-20 answers in {bundle_path.name} before spending "
          f"anything.")


if __name__ == "__main__":
    main()