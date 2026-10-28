# Router comparison: LLM extraction vs a free dictionary

Scored on the 259 held-out test queries that appear in both their published runs and our test split. Ground truth is the filing our qrels place the gold chunks in. This compares the routing component alone, so it is unaffected by the generator gap that dominates end-to-end scores.

| router | cost per query | coverage | precision | routed & correct |
|---|---|---|---|---|
| HDRR LLM router (hybrid) | 1 LLM call | 96.1% | 98.8% | 95.0% |
| ours: regex only | none | 78.0% | 100.0% | 78.0% |
| ours: SEC names + stoplist | none | 90.3% | 98.7% | 89.2% |

Where both routers committed to a filing (232 queries) they agreed 100.0% of the time. Our router committed on 2 queries theirs left unrouted; theirs committed on 17 ours left unrouted.

### HDRR LLM router (hybrid) - sample misroutes

| query | gold filing | chosen |
|---|---|---|
| CarMax FY23 vs. FY24 net cash increase investing strat. | RL | KMX |
| Seller experience on eBay directly impacts buyer confidence, which in turn affec | ETSY | EBAY |
| Fee model of Cboe incentivizes order flow liquidity for increased mkt share & pr | CME | CBOE |

### ours: SEC names + stoplist - sample misroutes

| query | gold filing | chosen |
|---|---|---|
| Revenue recognition: principal vs. agent criteria? | ARE | PFG |
| CarMax FY23 vs. FY24 net cash increase investing strat. | RL | KMX |
| Seller experience on eBay directly impacts buyer confidence, which in turn affec | ETSY | EBAY |

