# Offline diagnostics on run files

Test questions: 1088  |  ticker named: 853  |  not named: 235

## 1. Circularity check: is the routing gain an artifact?

If gold labels were ticker-dependent, ft_dense (which uses NO ticker
information) would score very differently on the two groups.

| System | group | n | NDCG@10 | 95% CI |
|---|---|---|---|---|
| ft_dense | ticker named | 853 | 0.1566 | [0.1373, 0.1786] |
| ft_dense | not named | 235 | 0.2954 | [0.2475, 0.3453] |
| small_dense | ticker named | 853 | 0.0600 | [0.0470, 0.0734] |
| small_dense | not named | 235 | 0.1417 | [0.1081, 0.1773] |
| ft_dense_ticker | ticker named | 853 | 0.6000 | [0.5783, 0.6213] |
| ft_dense_ticker | not named | 235 | 0.2954 | [0.2475, 0.3453] |

## 2. What did routing actually fix?

ft_dense failures split by cause. Routing can only fix the first.

- gold found in top-10: **329** (30.2%)
- missed, but the right company was in top-10 (routing cannot help): **186** (17.1%)
- missed, right company absent from top-10 entirely (routing fixes this): **573** (52.7%)

Mean fraction of ft_dense's top-10 that comes from the correct company: **0.144**

## 3. Where does reranking go wrong?

Gold chunks that dense retrieval placed in the top 10, and what reranking then did with them.

| reranked system | gold kept in top-10 | gold pushed out | gold pulled in |
|---|---|---|---|
| ft_dense_rerank@50 | 162 | 273 | 78 |
| ft_dense_rerank_base@50 | 332 | 103 | 113 |
| ft_dense_ticker_rerank@50 | 1082 | 271 | 185 |

## 4. Routing effect measured only on the queries it applies to

The headline +0.3468 is diluted by 235 unnamed queries where routing
falls back to plain dense retrieval. Restricted to the 853 queries
that actually get routed:

| System | NDCG@10 | Recall@100 | 95% CI (NDCG) |
|---|---|---|---|
| ft_dense | 0.1566 | 0.3745 | [0.1373, 0.1786] |
| ft_dense_ticker | 0.6000 | 0.9827 | [0.5783, 0.6213] |
| ft_dense_ticker_rerank@50 | 0.6599 | 0.9416 | [0.6383, 0.6799] |
