# FinRAG — project state

A retrieval-augmented generation pipeline over 497 S&P 500 10-K filings
(FinDER benchmark), running entirely on one RTX 4080 Laptop GPU (12.9 GB) with
no paid inference. Authors: Haroon M Bilal, Muhammad Bin Sarwar (UMT).

**The measurement phase is finished.** What remains is one GPU job and the
paper. Read this before proposing new experiments — most of the obvious ones
have already been run and ruled out.

## Where things stand

The paper (`paper/paper.tex`, builds with the bundled `IEEEtran.cls`) argues
that an end-to-end LLM-judged score cannot tell you where a RAG system's
quality comes from, and measures exactly that against a published baseline.

Headline numbers, all on filings never seen in training:

| | |
|---|---|
| FinRAG vs HDRR (their GPT-4.1 judge, their rubric) | 6.02 vs 7.63, 95% CI [−2.08, −1.13] |
| FinRAG with gold evidence (ceiling) | 6.70 |
| Gap decomposition | 0.67 retrieval + 0.93 generator |
| Retrieval, full pipeline | NDCG@10 0.5994 |
| Routing, ours vs their LLM router | 89.2% vs 95.0% correct, 100% agreement where both commit |
| Judge replication against their own answers | 90% exact, 100% within 1 point, r = 0.993 |

We lose on answer quality and report it as a loss. Their answers come from a
frontier API model; ours from a 4-bit 7B on a laptop. Do not try to reframe
this into a win.

## The one job left

The three-model generation table was produced on the **old** pipeline (ticker
regex, context depth 3) while the headline comparison uses the **new** one
(SEC-name routing, depth 10). One paper currently describes two systems.

```powershell
python -u src\generate\run_generation_eval.py Qwen/Qwen2.5-3B-Instruct 0 v2
python -u src\generate\run_generation_eval.py meta-llama/Llama-3.2-3B-Instruct 0 v2
python -u src\generate\run_generation_eval.py mistralai/Mistral-7B-Instruct-v0.3 0 v2
```

The second argument is the question limit; 0 means no limit.
About an hour each. Crash-safe — predictions flush to disk and reload, so a
restart costs minutes. Writes `results/generation_<tag>_v4_v2.{md,json}`,
leaving the v1 files intact. Then update Table IV in `paper/paper.tex` and say
in the caption which pipeline it describes.

## Things already ruled out — do not redo

- **Sparse/RRF fusion.** Hurts once the embedder is domain-adapted: −0.0189
  NDCG@10, CI [−0.0314, −0.0056].
- **Reranking before routing.** Sharply harmful, −0.1021. The same reranker
  after routing is +0.0075 and not significant. Component value is positional.
- **Context depth beyond 10.** Depth 3 → 10 gave +0.28 judged score; 10 → 20
  gave little more, and truncation rose. Depth 10 is the operating point.
- **Retraining the adapter for deeper context.** Checked: already-covered
  queries moved −0.12 (noise) when context went 3 → 10, so there is no
  train/test mismatch to fix.
- **A 14B scaling study.** Deliberately dropped. Several days on Kaggle T4s to
  add one point to a curve that would not change the conclusion.

## Things worth doing, in order

1. **Judge calibration at the low end** (~$0.20, minutes). The replication used
   40 of HDRR's answers, which average 7.95. It establishes agreement near 8,
   not near our own 6.02. Re-check on their low-scoring answers.
2. **RAGAS Context Recall.** HDRR reports no retrieval metrics, but the FinDER
   paper itself reports Context Recall: BM25 11.68, GTE 17.83, mE5 17.36,
   E5-Mistral 25.95. Computing it would put our retrieval on a published table
   instead of standing alone. This is the highest-value remaining item.
3. **Judge the depth-20 answers** (~$0.60). Already generated and cached;
   completes the context-depth ablation. Lowest priority.

## Layout

```
src/data/       corpus prep, labelling (prepare_finder_v4), split
src/train/      negative mining + embedder/reranker fine-tuning
src/generate/   QA data prep, QLoRA training, generation eval
src/eval/       ablation, HDRR comparison, routing, diagnostics
results/        metrics, predictions, TREC run files, figures
paper/          paper.tex, paper.pdf, figures/, IEEEtran.cls
external/hdrr/  cloned comparison repo (gitignored, re-clone if absent)
data/           corpus + labels (gitignored, ~5 GB)
checkpoints/    fine-tuned adapters (gitignored)
```

Key scripts in `src/eval/`:

- `run_ablation_split.py` — the 12-config retrieval ablation; writes TREC run
  files so every later metric question is offline re-scoring, not another
  multi-hour GPU run.
- `build_ticker_map.py` — builds the SEC name dictionary and **measures routing
  accuracy against qrels**, old rule vs new.
- `reroute_and_rerank.py` — rebuilds the run file for queries whose routing
  changed only; minutes, not hours.
- `build_hdrr_comparison.py` → `judge_openai.py` → `compare_hdrr.py` — generate
  answers, score with HDRR's judge, paired bootstrap.
- `diagnose_scores.py`, `delta_by_depth.py`, `qrels_contamination.py` — free
  CPU-only diagnostics that explain *why* a number moved.

## Conventions that matter

- **Never re-run retrieval to answer a metric question.** The TREC run files in
  `results/runs/` hold the top-100 per query; re-score them offline.
- **Chunk lengths are measured in the model's own tokenizer**, not in words. A
  word-defined chunk silently truncates at the encoder's 512-token limit.
- **Splits are at the filing level**, never the question level. Many questions
  target the same filing, so a question-level split leaks.
- **Paired bootstraps share resampling indices across systems.** Independent
  per-system bootstraps overstate uncertainty on a difference.
- **Corpus embedding caches carry a manifest** (model, chunk count, fingerprint)
  and hard-fail on mismatch. A stale cache silently indexing the wrong corpus
  cost days once already.
- Generation scripts cache predictions to disk and reload on restart. If every
  prediction is cached the model is never loaded and the run is a rescore.

## History worth knowing

Eight stacked defects were found and fixed, each hidden by the one above it:
evaluation leakage, 99.2% chunk truncation, cross-filing label contamination,
graded labels collapsing to binary at the split, a chunk-size-dependent
threshold, a positional generator split, a 1024-token training budget that fit
zero examples, and 88% refusal footers in the QA training data. Results before
these fixes are not comparable to current ones; the pre-fix draft claimed
MRR@10 0.260 and "we beat the state of the art", both false.

The ticker regex that routed queries also ran as an override in the labelling
step, so the labels were audited for the same acronym collision: zero affected
questions on test, 166 (3.9%) on train whose gold chunks match the query text
as well as any other question's. See `results/qrels_contamination.md`.
