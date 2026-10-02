# Generation eval: mistralai/Mistral-7B-Instruct-v0.3

Held-out test questions from filings never seen in training. `gold` = evidence chunks (ceiling); `retrieved` = top-3 from the deployed pipeline. Footer lines are stripped before ROUGE/BERTScore. Base conditions are subsampled to 300 questions (base emits a parseable footer on <1% of questions, so its numeric score is structurally zero).

| condition | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore-F1 |
|---|---|---|---|---|---|---|
| gold_base | 300 | 0.3740 | 0.1171 | 0.2033 | 0.0474 | 0.8576 |
| gold_ft | 1044 | 0.4740 | 0.1748 | 0.2518 | 0.0899 | 0.8783 |
| retrieved_base | 300 | 0.3562 | 0.1016 | 0.1895 | 0.0370 | 0.8521 |
| retrieved_ft | 1044 | 0.4357 | 0.1521 | 0.2318 | 0.0751 | 0.8712 |

## Structured output

| condition | strict fmt | parsed | abstain | truncated |
|---|---|---|---|---|
| gold_base | 0.950 | 0.000 | 0.000 | 0.007 |
| gold_ft | 1.000 | 0.999 | 0.041 | 0.000 |
| retrieved_base | 0.953 | 0.000 | 0.000 | 0.003 |
| retrieved_ft | 0.998 | 0.998 | 0.060 | 0.002 |

## Numeric exact-match (trusted subset, rounding-aware)

A prediction is correct if it equals the gold value rounded to the precision the prediction states.

| condition | n | EM | EM scale-lenient | correct | scale err | wrong | abstained | malformed | truncated |
|---|---|---|---|---|---|---|---|---|---|
| gold_base | 31 | 0.000 | 0.000 | 0 | 0 | 0 | 0 | 31 | 0 |
| gold_ft | 121 | 0.140 | 0.157 | 17 | 2 | 72 | 29 | 1 | 0 |
| retrieved_base | 31 | 0.000 | 0.000 | 0 | 0 | 0 | 0 | 30 | 1 |
| retrieved_ft | 121 | 0.099 | 0.099 | 12 | 0 | 66 | 43 | 0 | 0 |
