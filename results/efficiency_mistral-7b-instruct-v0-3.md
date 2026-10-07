# Efficiency: mistralai/Mistral-7B-Instruct-v0.3

GPU: NVIDIA GeForce RTX 4080 Laptop GPU (12.9 GB). 50 test queries, 80% ticker-routed. All three models (embedder, reranker, generator) are loaded SIMULTANEOUSLY, as they would be in deployment - per-component figures understate the requirement. Corpus vectors (181867 chunks) are memory-mapped, not held in VRAM.

## Peak VRAM

| stage | GB |
|---|---|
| + embedder (33M, fp32) | 0.13 |
| + reranker (278M, fp32) | 0.13 |
| + generator (4-bit QLoRA) | 4.44 |
| **peak under load** | **7.27** |

Headroom on a 13 GB card: 5.61 GB.

## Latency per query (seconds)

| stage | mean | median | p90 |
|---|---|---|---|
| embed | 0.005 | 0.006 | 0.010 |
| route | 0.005 | 0.000 | 0.020 |
| rerank | 0.477 | 0.469 | 0.525 |
| generate | 17.778 | 13.747 | 36.268 |
| end_to_end | 18.264 | 14.286 | 36.710 |

## Generation throughput

- **11.4 tokens/second** (greedy, batch size 1)
- mean 203 new tokens per answer
- model load time: embedder 1.2s, reranker 0.4s, generator 15.7s (one-off, not per query)
