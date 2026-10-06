# Generation eval: meta-llama/Llama-3.2-3B-Instruct

Held-out test questions from filings never seen in training. `gold` = evidence chunks (ceiling); `retrieved` = top-3 from the deployed pipeline. Footer lines are stripped before ROUGE/BERTScore. Base conditions are subsampled to 300 questions (base emits a parseable footer on <1% of questions, so its numeric score is structurally zero).

| condition | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore-F1 |
|---|---|---|---|---|---|---|
| gold_base | 300 | 0.3356 | 0.1069 | 0.1888 | 0.0361 | 0.8535 |
| gold_ft | 1044 | 0.4296 | 0.1547 | 0.2312 | 0.0751 | 0.8712 |
| retrieved_base | 300 | 0.3112 | 0.0877 | 0.1717 | 0.0279 | 0.8468 |
| retrieved_ft | 1044 | 0.3946 | 0.1340 | 0.2119 | 0.0645 | 0.8644 |

## Structured output

| condition | strict fmt | parsed | abstain | truncated |
|---|---|---|---|---|
| gold_base | 0.140 | 0.030 | 0.000 | 0.000 |
| gold_ft | 1.000 | 0.999 | 0.082 | 0.000 |
| retrieved_base | 0.103 | 0.013 | 0.000 | 0.000 |
| retrieved_ft | 1.000 | 1.000 | 0.122 | 0.000 |

## Numeric exact-match (trusted subset, rounding-aware)

A prediction is correct if it equals the gold value rounded to the precision the prediction states.

| condition | n | EM | EM scale-lenient | correct | scale err | wrong | abstained | malformed | truncated |
|---|---|---|---|---|---|---|---|---|---|
| gold_base | 31 | 0.032 | 0.065 | 1 | 1 | 4 | 0 | 25 | 0 |
| gold_ft | 121 | 0.091 | 0.107 | 11 | 2 | 66 | 41 | 1 | 0 |
| retrieved_base | 31 | 0.032 | 0.032 | 1 | 0 | 2 | 0 | 28 | 0 |
| retrieved_ft | 121 | 0.083 | 0.099 | 10 | 2 | 59 | 50 | 0 | 0 |
