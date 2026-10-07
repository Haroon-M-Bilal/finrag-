# Efficiency: Qwen/Qwen2.5-3B-Instruct

GPU: NVIDIA GeForce RTX 4080 Laptop GPU (12.9 GB). 50 test queries, 80% ticker-routed. All three models (embedder, reranker, generator) are loaded SIMULTANEOUSLY, as they would be in deployment - per-component figures understate the requirement. Corpus vectors (181867 chunks) are memory-mapped, not held in VRAM.

## Peak VRAM

| stage | GB |
|---|---|
| + embedder (33M, fp32) | 0.13 |
| + reranker (278M, fp32) | 0.13 |
| + generator (4-bit QLoRA) | 2.91 |
| **peak under load** | **6.30** |

Headroom on a 13 GB card: 6.58 GB.

## Latency per query (seconds)

| stage | mean | median | p90 |
|---|---|---|---|
| embed | 0.005 | 0.002 | 0.010 |
| route | 0.005 | 0.000 | 0.020 |
| rerank | 0.518 | 0.507 | 0.586 |
| generate | 18.190 | 14.410 | 35.897 |
| end_to_end | 18.718 | 14.968 | 36.361 |

## Generation throughput

- **9.5 tokens/second** (greedy, batch size 1)
- mean 174 new tokens per answer
- model load time: embedder 1.0s, reranker 0.4s, generator 7.8s (one-off, not per query)
