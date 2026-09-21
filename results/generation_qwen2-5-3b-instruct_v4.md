# Generation eval: Qwen/Qwen2.5-3B-Instruct

1044 held-out test questions from filings never seen in training. `gold` = evidence chunks (ceiling); `retrieved` = top-3 from the deployed pipeline. Footer lines are stripped before ROUGE/BERTScore.

| condition | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore-F1 |
|---|---|---|---|---|---|
| gold_base | 0.2978 | 0.0979 | 0.1768 | 0.0294 | 0.8539 |
| gold_ft | 0.4208 | 0.1443 | 0.2229 | 0.0696 | 0.8704 |
| retrieved_base | 0.2915 | 0.0875 | 0.1697 | 0.0263 | 0.8509 |
| retrieved_ft | 0.3858 | 0.1258 | 0.2041 | 0.0593 | 0.8626 |

## Structured output

| condition | strict fmt | parsed | abstain | truncated |
|---|---|---|---|---|
| gold_base | 0.919 | 0.003 | 0.000 | 0.000 |
| gold_ft | 0.999 | 0.999 | 0.104 | 0.001 |
| retrieved_base | 0.888 | 0.002 | 0.000 | 0.003 |
| retrieved_ft | 0.999 | 0.999 | 0.148 | 0.001 |

## Numeric exact-match (trusted subset, rounding-aware)

A prediction is correct if it equals the gold value rounded to the precision the prediction states.

| condition | n | EM | EM scale-lenient | correct | scale err | wrong | abstained | malformed | truncated |
|---|---|---|---|---|---|---|---|---|---|
| gold_base | 121 | 0.000 | 0.000 | 0 | 0 | 3 | 0 | 118 | 0 |
| gold_ft | 121 | 0.190 | 0.198 | 23 | 1 | 60 | 37 | 0 | 0 |
| retrieved_base | 121 | 0.000 | 0.000 | 0 | 0 | 1 | 0 | 120 | 0 |
| retrieved_ft | 121 | 0.107 | 0.107 | 13 | 0 | 54 | 54 | 0 | 0 |
