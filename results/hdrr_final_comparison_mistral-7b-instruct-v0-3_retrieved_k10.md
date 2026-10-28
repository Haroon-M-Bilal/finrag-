# FinRAG vs Cheng et al. (2026) - paired comparison

All systems scored by the same judge (`gpt-4.1`, temperature 0) on the same 258 FinDER queries. Those queries are the intersection of their published 300-query splits with our 98 held-out test filings, so none of them touched any filing our retriever, reranker or generator was trained on.

Our judge reproduces theirs: on 40 of their own answers, re-scoring agreed exactly 90% of the time and within one point 100% of the time (mean difference +0.05, r = 0.993).

## Per-system results on the shared queries

| System | n | mean | failure (<=1) | correct (>=8) |
|---|---|---|---|---|
| **FinRAG (ours)** | 258 | **5.82** | 17.8% | 44.2% |
| HDRR (their best) | 258 | 7.63 | 5.8% | 66.3% |
| CBR + LLM routing | 258 | 7.44 | 7.0% | 65.1% |
| CBR + metadata | 258 | 6.84 | 12.0% | 58.1% |
| Unnamed baseline | 258 | 6.61 | 7.0% | 45.0% |
| Agentic RAG | 258 | 6.27 | 19.4% | 50.4% |

## Paired bootstrap (ours minus theirs, 95% CI, 10,000 resamples)

A CI excluding zero is a real difference. Win/loss counts the queries where our score was strictly higher or lower.

| vs | delta | 95% CI | significant | win | loss | tie |
|---|---|---|---|---|---|---|
| Agentic RAG | -0.45 | [-0.99, +0.10] | no | 96 | 123 | 39 |
| Unnamed baseline | -0.79 | [-1.26, -0.32] | **yes** | 97 | 111 | 50 |
| CBR + metadata | -1.03 | [-1.52, -0.53] | **yes** | 68 | 136 | 54 |
| CBR + LLM routing | -1.62 | [-2.11, -1.12] | **yes** | 62 | 148 | 48 |
| HDRR (their best) | -1.81 | [-2.29, -1.33] | **yes** | 53 | 152 | 53 |

## Secondary: excluding our abstentions

Their rubric instructs the judge to score an answer 1-3 when it says the information cannot be found, regardless of whether refusing was correct. Our generator is calibrated to abstain, and does so on 16 of 258 queries (6.2%). Removing them shows how much of the gap is that rubric clause rather than a pipeline difference. **The headline result is the table above; this one is an explanation, not a competing number.**

FinRAG excluding abstentions: mean 6.10 over 242 queries (+0.29).

| vs | delta | 95% CI | significant |
|---|---|---|---|
| Agentic RAG | -0.21 | [-0.78, +0.35] | no |
| Unnamed baseline | -0.62 | [-1.09, -0.15] | **yes** |
| CBR + metadata | -0.82 | [-1.35, -0.32] | **yes** |
| CBR + LLM routing | -1.39 | [-1.87, -0.90] | **yes** |
| HDRR (their best) | -1.57 | [-2.05, -1.10] | **yes** |

## Bottom line

HDRR scores 1.81 above FinRAG (95% CI [-2.29, -1.33]). The difference is real and should be reported as such; the contribution to argue is efficiency and cost, not answer quality.
