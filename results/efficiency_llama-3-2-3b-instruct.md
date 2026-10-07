# Efficiency: meta-llama/Llama-3.2-3B-Instruct

GPU: NVIDIA GeForce RTX 4080 Laptop GPU (12.9 GB). 50 test queries, 80% ticker-routed. All three models (embedder, reranker, generator) are loaded SIMULTANEOUSLY, as they would be in deployment - per-component figures understate the requirement. Corpus vectors (181867 chunks) are memory-mapped, not held in VRAM.

## Peak VRAM

| stage | GB |
|---|---|
| + embedder (33M, fp32) | 0.13 |
| + reranker (278M, fp32) | 0.13 |
| + generator (4-bit QLoRA) | 2.49 |
| **peak under load** | **5.32** |

Headroom on a 13 GB card: 7.56 GB.

## Latency per query (seconds)

| stage | mean | median | p90 |
|---|---|---|---|
| embed | 0.006 | 0.008 | 0.010 |
| route | 0.004 | 0.000 | 0.015 |
| rerank | 0.492 | 0.489 | 0.553 |
| generate | 14.270 | 11.581 | 27.954 |
| end_to_end | 14.773 | 12.044 | 28.450 |

## Generation throughput

- **13.2 tokens/second** (greedy, batch size 1)
- mean 188 new tokens per answer
- model load time: embedder 0.9s, reranker 0.5s, generator 5.9s (one-off, not per query)
