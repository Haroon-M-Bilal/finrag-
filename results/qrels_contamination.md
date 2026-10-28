# qrels contamination from the ticker/acronym collision

`prepare_finder_v4.py` resolved questions to filings with the same regex the router used, so questions using CF, NI, IR, IP and similar as ordinary financial vocabulary were force-labelled to those companies and then had gold chunks selected from inside the wrong filing.

## test

1088 labelled questions across 497 filings (expected 2.2 each).

| filing | questions | x expected | stoplisted |
|---|---|---|---|
| CINF | 16 | 7.3x | no |
| CME | 15 | 6.9x | no |
| PANW | 14 | 6.4x | no |
| APA | 14 | 6.4x | no |
| CMCSA | 14 | 6.4x | no |
| JKHY | 14 | 6.4x | no |
| CTSH | 14 | 6.4x | no |
| MSFT | 14 | 6.4x | no |
| ADP | 14 | 6.4x | no |
| SBUX | 14 | 6.4x | no |
| ISRG | 14 | 6.4x | no |
| MCHP | 14 | 6.4x | no |

Excess over expectation on stoplisted tickers: **0 questions (0.0%)**. Questions matching the misfire condition exactly (labelled to a stoplisted ticker, containing that token, naming no other): **0 (0.0%)**.

## train

4215 labelled questions across 497 filings (expected 8.5 each).

| filing | questions | x expected | stoplisted |
|---|---|---|---|
| EXC | 25 | 2.9x | yes |
| TRV | 23 | 2.7x | yes |
| CF | 22 | 2.6x | yes |
| PG | 21 | 2.5x | yes |
| CRWD | 15 | 1.8x | no |
| ADI | 15 | 1.8x | no |
| FTNT | 15 | 1.8x | no |
| TMO | 15 | 1.8x | no |
| STX | 15 | 1.8x | no |
| EQIX | 15 | 1.8x | no |
| AAPL | 15 | 1.8x | no |
| CSX | 15 | 1.8x | no |

Excess over expectation on stoplisted tickers: **91 questions (2.2%)**. Questions matching the misfire condition exactly (labelled to a stoplisted ticker, containing that token, naming no other): **166 (3.9%)**.

Query-term coverage by the selected gold chunks is 0.477 for suspects against 0.450 for the rest.

