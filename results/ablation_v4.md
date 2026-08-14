# Ablation on held-out TEST filings (1088 questions)

Corpus 181867 chunks, data\finder\processed_v4. NDCG uses LINEAR gain (g / log2(r+1)); ndcg_exp uses (2^g - 1). Graded qrels: grade 2 = primary evidence chunk, grade 1 = supporting. Ticker named in 853/1088 queries (78.4%).

| System | ndcg@10 | ndcg@20 | mrr@10 | recall@10 | recall@100 | NDCG@10 95% CI |
|---|---|---|---|---|---|---|
| bm25 | 0.0359 | 0.0431 | 0.0403 | 0.0613 | 0.1742 | [0.0284, 0.0445] |
| small_dense | 0.0777 | 0.0864 | 0.0807 | 0.1134 | 0.2256 | [0.0647, 0.0917] |
| base_dense | 0.0681 | 0.0785 | 0.0727 | 0.1030 | 0.2304 | [0.0569, 0.0803] |
| ft_dense | 0.1866 | 0.1984 | 0.1994 | 0.2420 | 0.4032 | [0.1677, 0.2062] |
| ft_hybrid | 0.1678 | 0.1819 | 0.1739 | 0.2451 | 0.4033 | [0.1496, 0.1868] |
| ft_hybrid_rerank@50 | 0.1046 | 0.1271 | 0.0997 | 0.1780 | 0.3517 | [0.0920, 0.1191] |
| ft_dense_rerank@50 | 0.0845 | 0.1021 | 0.0817 | 0.1396 | 0.3414 | [0.0721, 0.0969] |
| ft_dense_rerank@100 | 0.0719 | 0.0906 | 0.0731 | 0.1080 | 0.4032 | [0.0603, 0.0836] |
| ft_dense_rerank@200 | 0.0607 | 0.0743 | 0.0627 | 0.0882 | 0.3393 | [0.0503, 0.0713] |
| ft_dense_rerank_base@50 | 0.1719 | 0.1853 | 0.1831 | 0.2490 | 0.3414 | [0.1551, 0.1903] |
| ft_dense_ticker | 0.5342 | 0.5549 | 0.5354 | 0.7021 | 0.8801 | [0.5120, 0.5567] |
| ft_dense_ticker_rerank@50 | 0.5414 | 0.5632 | 0.5550 | 0.6706 | 0.8368 | [0.5179, 0.5640] |

## Paired bootstrap on NDCG@10 (same resamples across systems)

| A | B | A-B | 95% CI | significant |
|---|---|---|---|---|
| ft_dense | small_dense | +0.1091 | [+0.0903, +0.1287] | yes |
| ft_dense | base_dense | +0.1186 | [+0.0980, +0.1376] | yes |
| ft_dense | bm25 | +0.1508 | [+0.1306, +0.1709] | yes |
| ft_hybrid | ft_dense | -0.0189 | [-0.0314, -0.0056] | yes |
| ft_dense_rerank@50 | ft_dense | -0.1021 | [-0.1211, -0.0841] | yes |
| ft_dense_rerank@200 | ft_dense_rerank@50 | -0.0238 | [-0.0325, -0.0151] | yes |
| ft_dense_rerank@50 | ft_dense_rerank_base@50 | -0.0872 | [-0.1037, -0.0714] | yes |
| ft_dense_ticker | ft_dense | +0.3468 | [+0.3269, +0.3691] | yes |
| ft_dense_ticker_rerank@50 | ft_dense_ticker | +0.0075 | [-0.0156, +0.0297] | no |
