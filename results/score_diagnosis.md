# Why the scores are bimodal

Joining each per-query judge score to whether gold evidence actually reached the generator's top-3 context.

## Score by retrieval outcome

| context | n | mean | median | <=2 | >=8 |
|---|---|---|---|---|---|
| at least one gold chunk | 162 | 6.44 | 8.0 | 21.6% | 54.9% |
| ALL gold chunks | 109 | 6.77 | 8.0 | 18.3% | 60.6% |
| no gold chunk | 96 | 4.02 | 2.0 | 54.2% | 19.8% |

## Their hybrid on the same split

Their system does not use our retrieval, so this is a control: if their scores also collapse on the queries where OUR retrieval missed, those queries are simply hard. If their scores hold up, the gap is ours to fix.

| our context | their mean |
|---|---|
| gold reached | 7.54 |
| gold missed | 7.81 |

## Where the missed gold actually sits

96 queries had no gold in the top-3. Rank of the first gold chunk for those:

| rank of first gold | queries | recoverable by |
|---|---|---|
| 4-5 | 10 | TOPK_CTX=5 |
| 6-10 | 19 | TOPK_CTX=10 |
| 11-20 | 22 | TOPK_CTX=20 |
| 21-50 | 17 | deeper reranking |
| 51-100 | 0 | retrieval, not context |
| not in top-100 | 28 | retrieval failure |

## Expected payoff of more context

Raising the generator's context from top-3 to top-10 would put gold in front of it for **29 additional queries**; top-20 would reach **51**. Those queries currently average 4.02. If they behaved like the queries that do get gold (6.44), the overall mean would move from 5.54 to roughly 5.81 at top-10.

That is an upper bound, not a promise: the model was fine-tuned on roughly two gold chunks per example, so feeding it ten is a train/test mismatch that may cost some accuracy on queries it already answers well. It has to be measured, not assumed.

