# Paired bootstrap on the routed subset (853 queries)

Restricted to test queries that name exactly one ticker, i.e. the queries where routing actually applies. The full-set delta for reranking (+0.0075, CI spanning zero) was diluted by 235 unrouted queries where both systems fall back to identical plain dense retrieval.

| System | ndcg@10 | mrr@10 | recall@10 | recall@100 | NDCG@10 95% CI |
|---|---|---|---|---|---|
| ft_dense | 0.1566 | 0.1667 | 0.2054 | 0.3745 | [0.1373, 0.1786] |
| small_dense | 0.0600 | 0.0600 | 0.0897 | 0.1955 | [0.0470, 0.0734] |
| base_dense | 0.0536 | 0.0540 | 0.0819 | 0.2025 | [0.0423, 0.0661] |
| ft_dense_ticker | 0.6000 | 0.5952 | 0.7922 | 0.9827 | [0.5783, 0.6213] |
| ft_dense_ticker_rerank@50 | 0.6599 | 0.6773 | 0.8042 | 0.9416 | [0.6383, 0.6799] |

## Paired deltas (shared resamples)

| A | B | metric | A-B | 95% CI | significant |
|---|---|---|---|---|---|
| ft_dense_ticker | ft_dense | ndcg@10 | +0.4433 | [+0.4187, +0.4656] | yes |
| ft_dense_ticker | ft_dense | mrr@10 | +0.4289 | [+0.4021, +0.4550] | yes |
| ft_dense_ticker | ft_dense | recall@10 | +0.5867 | [+0.5561, +0.6151] | yes |
| ft_dense_ticker_rerank@50 | ft_dense_ticker | ndcg@10 | +0.0595 | [+0.0370, +0.0851] | yes |
| ft_dense_ticker_rerank@50 | ft_dense_ticker | mrr@10 | +0.0811 | [+0.0511, +0.1129] | yes |
| ft_dense_ticker_rerank@50 | ft_dense_ticker | recall@10 | +0.0117 | [-0.0104, +0.0342] | no |
| ft_dense_ticker_rerank@50 | ft_dense | ndcg@10 | +0.5028 | [+0.4758, +0.5292] | yes |
| ft_dense_ticker_rerank@50 | ft_dense | mrr@10 | +0.5100 | [+0.4785, +0.5413] | yes |
| ft_dense_ticker_rerank@50 | ft_dense | recall@10 | +0.5984 | [+0.5665, +0.6287] | yes |
