# FinRAG vs Cheng et al. (2026) - paired comparison

All systems scored by the same judge (`gpt-4.1`, temperature 0) on the same 258 FinDER queries. Those queries are the intersection of their published 300-query splits with our 98 held-out test filings, so none of them touched any filing our retriever, reranker or generator was trained on.

Our judge reproduces theirs: on 40 of their own answers, re-scoring agreed exactly 90% of the time and within one point 100% of the time (mean difference +0.05, r = 0.993).

## Per-system results on the shared queries

| System | n | mean | failure (<=1) | correct (>=8) |
|---|---|---|---|---|
| **FinRAG (ours)** | 258 | **6.70** | 9.3% | 56.2% |
| HDRR (their best) | 258 | 7.63 | 5.8% | 66.3% |
| CBR + LLM routing | 258 | 7.44 | 7.0% | 65.1% |
| CBR + metadata | 258 | 6.84 | 12.0% | 58.1% |
| Unnamed baseline | 258 | 6.61 | 7.0% | 45.0% |
| Agentic RAG | 258 | 6.27 | 19.4% | 50.4% |

## Paired bootstrap (ours minus theirs, 95% CI, 10,000 resamples)

A CI excluding zero is a real difference. Win/loss counts the queries where our score was strictly higher or lower.

| vs | delta | 95% CI | significant | win | loss | tie |
|---|---|---|---|---|---|---|
| Agentic RAG | +0.43 | [-0.09, +0.95] | no | 113 | 93 | 52 |
| Unnamed baseline | +0.09 | [-0.35, +0.52] | no | 127 | 85 | 46 |
| CBR + metadata | -0.15 | [-0.67, +0.37] | no | 89 | 108 | 61 |
| CBR + LLM routing | -0.74 | [-1.22, -0.28] | **yes** | 74 | 129 | 55 |
| HDRR (their best) | -0.93 | [-1.37, -0.49] | **yes** | 72 | 123 | 63 |

## Secondary: excluding our abstentions

Their rubric instructs the judge to score an answer 1-3 when it says the information cannot be found, regardless of whether refusing was correct. Our generator is calibrated to abstain, and does so on 11 of 258 queries (4.3%). Removing them shows how much of the gap is that rubric clause rather than a pipeline difference. **The headline result is the table above; this one is an explanation, not a competing number.**

FinRAG excluding abstentions: mean 6.95 over 247 queries (+0.25).

| vs | delta | 95% CI | significant |
|---|---|---|---|
| Agentic RAG | +0.74 | [+0.25, +1.24] | **yes** |
| Unnamed baseline | +0.33 | [-0.08, +0.76] | no |
| CBR + metadata | +0.13 | [-0.38, +0.63] | no |
| CBR + LLM routing | -0.52 | [-0.98, -0.06] | **yes** |
| HDRR (their best) | -0.67 | [-1.09, -0.24] | **yes** |

## Bottom line

HDRR scores 0.93 above FinRAG (95% CI [-1.37, -0.49]). The difference is real and should be reported as such; the contribution to argue is efficiency and cost, not answer quality.
