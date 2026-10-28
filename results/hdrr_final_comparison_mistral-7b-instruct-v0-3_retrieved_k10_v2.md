# FinRAG vs Cheng et al. (2026) - paired comparison

All systems scored by the same judge (`gpt-4.1`, temperature 0) on the same 258 FinDER queries. Those queries are the intersection of their published 300-query splits with our 98 held-out test filings, so none of them touched any filing our retriever, reranker or generator was trained on.

Our judge reproduces theirs: on 40 of their own answers, re-scoring agreed exactly 90% of the time and within one point 100% of the time (mean difference +0.05, r = 0.993).

## Per-system results on the shared queries

| System | n | mean | failure (<=1) | correct (>=8) |
|---|---|---|---|---|
| **FinRAG (ours)** | 258 | **6.02** | 17.1% | 48.1% |
| HDRR (their best) | 258 | 7.63 | 5.8% | 66.3% |
| CBR + LLM routing | 258 | 7.44 | 7.0% | 65.1% |
| CBR + metadata | 258 | 6.84 | 12.0% | 58.1% |
| Unnamed baseline | 258 | 6.61 | 7.0% | 45.0% |
| Agentic RAG | 258 | 6.27 | 19.4% | 50.4% |

## Paired bootstrap (ours minus theirs, 95% CI, 10,000 resamples)

A CI excluding zero is a real difference. Win/loss counts the queries where our score was strictly higher or lower.

| vs | delta | 95% CI | significant | win | loss | tie |
|---|---|---|---|---|---|---|
| Agentic RAG | -0.25 | [-0.78, +0.30] | no | 101 | 118 | 39 |
| Unnamed baseline | -0.59 | [-1.04, -0.12] | **yes** | 102 | 105 | 51 |
| CBR + metadata | -0.82 | [-1.32, -0.32] | **yes** | 75 | 133 | 50 |
| CBR + LLM routing | -1.41 | [-1.91, -0.92] | **yes** | 69 | 141 | 48 |
| HDRR (their best) | -1.61 | [-2.08, -1.13] | **yes** | 58 | 145 | 55 |

## Secondary: excluding our abstentions

Their rubric instructs the judge to score an answer 1-3 when it says the information cannot be found, regardless of whether refusing was correct. Our generator is calibrated to abstain, and does so on 17 of 258 queries (6.6%). Removing them shows how much of the gap is that rubric clause rather than a pipeline difference. **The headline result is the table above; this one is an explanation, not a competing number.**

FinRAG excluding abstentions: mean 6.34 over 241 queries (+0.32).

| vs | delta | 95% CI | significant |
|---|---|---|---|
| Agentic RAG | +0.02 | [-0.53, +0.57] | no |
| Unnamed baseline | -0.39 | [-0.85, +0.07] | no |
| CBR + metadata | -0.59 | [-1.09, -0.08] | **yes** |
| CBR + LLM routing | -1.16 | [-1.64, -0.66] | **yes** |
| HDRR (their best) | -1.34 | [-1.81, -0.87] | **yes** |

## Bottom line

HDRR scores 1.61 above FinRAG (95% CI [-2.08, -1.13]). The difference is real and should be reported as such; the contribution to argue is efficiency and cost, not answer quality.
