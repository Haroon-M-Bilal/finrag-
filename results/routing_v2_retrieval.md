# Routing v2: SEC company names + acronym stoplist

Held-out test questions (1088). Same pipeline as `ft_dense_ticker_rerank@50` - dense retrieval inside the routed filing, then the fine-tuned cross-encoder over the top 50 - with routing from SEC's public company-name mapping instead of a bare ticker regex. 135 queries changed routing (135 newly routed, 0 de-routed as acronym collisions, 0 re-targeted); the rest are byte-identical.

**Label integrity checked.** `prepare_finder_v4.py` resolved questions to filings using a ticker regex with the same acronym collision as the old router, so the labels were audited (`qrels_contamination.py`). On the test split, zero questions match the misfire condition, so these deltas are measured against sound labels. On train, 166 questions (3.9%) match it, but their gold chunks cover the query text as well as any other question's (0.477 vs 0.450), indicating genuine mentions rather than mislabelling.

| metric | regex routing | SEC-name routing | delta | 95% CI | sig |
|---|---|---|---|---|---|
| ndcg@10 | 0.5414 | 0.5994 | +0.0580 | [+0.0460, +0.0705] | **yes** |
| ndcg@20 | 0.5632 | 0.6202 | +0.0570 | [+0.0454, +0.0692] | **yes** |
| mrr@10 | 0.5550 | 0.6171 | +0.0621 | [+0.0490, +0.0761] | **yes** |
| recall@10 | 0.6706 | 0.7322 | +0.0616 | [+0.0482, +0.0757] | **yes** |
| recall@100 | 0.8368 | 0.8749 | +0.0381 | [+0.0277, +0.0493] | **yes** |

## The 135 queries whose routing changed

Everything else is unchanged by construction, so the aggregate delta above is this effect diluted across the full test set.

| metric | before | after | delta |
|---|---|---|---|
| ndcg@10 | 0.1337 | 0.6010 | +0.4674 |
| ndcg@20 | 0.1589 | 0.6182 | +0.4593 |
| mrr@10 | 0.1288 | 0.6290 | +0.5002 |
| recall@10 | 0.2298 | 0.7263 | +0.4965 |
| recall@100 | 0.5540 | 0.8613 | +0.3073 |
