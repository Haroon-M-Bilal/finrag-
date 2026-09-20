# Numeric extraction report (v4)

Numeric subset (FinDER's `type` field): **883** of 5703 (15.5%)

## Confidence tiers

| tier | n | share |
|---|---|---|
| anchored_verified | 626 | 70.9% |
| single_figure | 7 | 0.8% |
| anchored_only | 145 | 16.4% |
| unresolved | 105 | 11.9% |

**Trusted (verified + single): 633 (71.7%).**

Anchor rule used: POST-marker 763, PRE-marker 8. Both orders occur in FinDER; v3 handled only the first and left the second unresolved.

Verification requires the confirming operands to sit in the SAME CLAUSE as the target. v3 allowed any pair from the whole answer, which at 2% tolerance over ~10 operations matched almost always and so confirmed nothing.

## By question type

| type | n | verified | single | anchor only | unresolved |
|---|---|---|---|---|---|
| Compositional | 440 | 326 | 3 | 43 | 68 |
| Division | 128 | 99 | 2 | 9 | 18 |
| Multiplication | 121 | 60 | 0 | 59 | 2 |
| Subtract | 111 | 96 | 2 | 6 | 7 |
| Addition | 75 | 45 | 0 | 27 | 3 |
| Subtraction | 8 | 0 | 0 | 1 | 7 |

## Trusted samples

Each must show the STATED RESULT, not one of its inputs.

- `Subtract` `pre` → **111.5 | million | USD**  
  _The Data and Access Solutions revenue increased by $111.5 million from 2021 to 2023, calculated as 539.2 million minus 427.7 million._
- `Compositional` `post` → **0.18 | unit | PCT**  
  _The three building engineers represent approximately 0.18% of the total workforce (calculated as 3 ÷ 1,647 × 100 ≈ 0.18%). Although this is a very small portion of overall employee_
- `Compositional` `post` → **36.4 | unit | PCT**  
  _To calculate the revenue growth rates, we use the formula: [(Current Year Revenue - Previous Year Revenue) / Previous Year Revenue] x 100.

1. FY2022 to FY2023:
   - FY2022 Total R_
- `Addition` `post` → **1712 | million | USD**  
  _Calculation: The interest income for the period is $866 million and the other income (net) is $846 million, yielding a total non-operating income of $866 + $846 = $1,712 million. T_
- `Division` `post` → **3 | unit | COUNT**  
  _At the end of fiscal year 2024, NVIDIA had approximately 29,600 employees, of which 22,200 were engaged in R&D and 7,400 in sales, marketing, operations, and administration. This r_
- `Subtract` `post` → **5.38134e+06 | thousand | COUNT**  
  _To calculate the absolute gross profit for 2024, subtract the cost of sales from the revenue. Using the provided data:

Revenue (2024): 9,427,157 thousand
Cost of Sales (2024): 4,0_
- `Subtract` `post` → **1.97 | unit | USD**  
  _The basic net income per share increased from $2.26 in fiscal year 2022 to $4.23 in fiscal year 2024. This is calculated as $4.23 - $2.26 = $1.97. This substantial increase in EPS_
- `Subtract` `post` → **10611 | million | COUNT**  
  _To calculate the service gross profit for 2024, subtract the cost of service revenue from the service revenue. For Intuit Inc., the calculation is as follows:

Service Gross Profit_
- `Subtract` `post` → **8000 | unit | COUNT**  
  _Using the information provided, the headcount gap is calculated by subtracting the average number of seasonal employees (10,800) from the total employee count (18,800). The calcula_
- `Compositional` `post` → **9.1 | unit | PCT**  
  _Using the provided sales and net earnings data for Ross Stores, Inc., we can calculate the net profit margins for each fiscal year as follows:

1. Fiscal Year Ended February 3, 202_
- `Division` `post` → **5.67 | unit | COUNT**  
  _Step 1: Calculate the number of retail associates. Since 85% of 108,000 associates work in retail, retail associates = 0.85 * 108,000 = 91,800.
Step 2: Calculate the number of non‐_
- `Addition` `post` → **2.01592e+06 | unit | COUNT**  
  _For 2024, the cost of sales is $1,203,792 and the selling and administrative expenses are $812,128. Adding these together, the total operating expense is calculated as follows:

1,_

## Anchored but not locally confirmed

Inspect: correct picks the local check missed, plus genuine misses.

- `Multiplication` `post` → **7.98 | billion | USD**  
  _To find the Small Business & Self-Employed revenue for fiscal 2023, we can divide the fiscal 2024 revenue of $9.5 billion by the growth factor of 1.19. The calculation is as follow_
- `Multiplication` `post` → **0.104 | unit | COUNT**  
  _To calculate the EBIT multiplier, we take the reported EBIT for 2024 ($1,000 million) and divide it by the net sales for 2024 ($9,636 million). The calculation is as follows:

Mult_
- `Compositional` `post` → **347222 | unit | COUNT**  
  _To calculate the revenue per employee, you would use the formula:

  Revenue per Employee = (Total Annual Revenue in Dollars) / (Number of Employees).

For Campbell Soup Company, i_
- `Multiplication` `post` → **986.19 | million | COUNT**  
  _To compute the net income using the diluted metrics for fiscal year April 26, 2024, we multiply the diluted net income per share by the diluted shares outstanding: 4.63 × 213 ≈ 986_
- `Multiplication` `post` → **38000 | unit | COUNT**  
  _The total revenue can be calculated by multiplying the average revenue per employee (R dollars) by the total number of employees. Since there are 38,000 employees, the formula is:_
- `Multiplication` `post` → **7.04 | unit | COUNT**  
  _To find the multiplication factor, we divide the 2023 amount by the 2022 amount. That is, x = 542 / 77, which gives x ≈ 7.04. Therefore, the interest and other financial charges in_
- `Addition` `post` → **4.75465e+07 | unit | COUNT**  
  _To calculate the aggregate weighted-average diluted shares outstanding for the three-year period, we sum the diluted shares for each year:

• 2024: 15,408,095 shares
• 2023: 15,812_
- `Addition` `post` → **2208.1 | million | USD**  
  _For FY2023, American Electric Power Company reports net income attributable to AEP common shareholders of $2,208.1 million, with a weighted average of 518,903,682 basic shares outs_
- `Compositional` `post` → **17250 | unit | COUNT**  
  _To compute the total number of employees, you sum the employee counts from each subsidiary. Using the data provided:

• AEPSC: 6,736
• AEP Texas: 1,646
• APCo: 1,679
• I&M: 2,110
•_
- `Compositional` `post` → **24801 | unit | COUNT**  
  _For 2023, Amgen's gross profit margin can be calculated using the formula:

  (Product Sales + Other Revenues - Cost of Sales) / Total Revenues

Using the 2023 data:
  Product Sale_
- `Multiplication` `post` → **12000 | unit | COUNT**  
  _The question is answerable. To determine the total R&D expenditure, you multiply the number of R&D employees (12,000) by the R&D investment allocated per employee ($Y). Therefore,_
- `Multiplication` `post` → **5.68 | unit | PCT**  
  _To calculate the net profit margin for 2023, we use the formula: net profit margin = (net earnings / total operating revenues) * 100. For 2023, net earnings are 728,287 and total o_

## Unresolved

No anchor found; excluded.

- `Compositional` → **N/A**  
  _The trend in diluted EPS over the three years shows a significant improvement. In 2022, the reported diluted EPS was a loss of $0.90 per share, indicating negative earnings. In 202_
- `Compositional` → **N/A**  
  _To calculate the change in total personnel expense resulting from a year-over-year increase in employee count, you first need to know the difference in the number of employees betw_
- `Division` → **N/A**  
  _With roughly 54% of its workforce dedicated to engineering and a relatively low voluntary turnover rate of 8%, Analog Devices, Inc. appears to maintain a stable, technically profic_
- `Addition` → **N/A**  
  _Based on the provided reference, Nordson Corporation had approximately 8,000 employees as of October 31, 2024. No additional calculations are needed since the data explicitly state_
- `Subtract` → **N/A**  
  _The provided information only includes the current number of employees (approximately 11,800 as of April 26, 2024) and does not provide the previous year's employee count. Without_
- `Compositional` → **N/A**  
  _The provided data allows us to compare diluted EPS figures with and without the impact of discontinued operations over the three years. In 2023, discontinued operations are not rep_
- `Compositional` → **N/A**  
  _Between the fiscal years, Seagate turned around its operating performance considerably. In the fiscal year ended June 30, 2023, the company reported an operating loss of $342 milli_
- `Subtract` → **N/A**  
  _The current headcount is 51,000. Assuming that last year's headcount was W, the difference is calculated as 51,000 - W. This simple subtraction gives the net change in employees, w_
- `Division` → **N/A**  
  _Calculation: The ratio of non-U.S. employees to total employees is obtained by dividing 79,000 by 91,000, which is approximately 0.87 or 87%. This means that roughly 87% of the wor_
- `Subtraction` → **N/A**  
  _The number of full-time equivalent employees decreased from 19,319 in 2022 to 18,724 in 2023, a reduction of 595 employees. This reduction could suggest an effort to improve operat_
- `Compositional` → **N/A**  
  _Netflix's net income increased from $4,491,924 in 2022 to $5,407,990 in 2023. This represents an increase of $916,066, which is approximately a 20.4% rise in net income, indicating_
- `Subtraction` → **N/A**  
  _For 2023, Netflix's basic EPS is $12.25 and its diluted EPS is $12.03, showing a dilution of $0.22 per share. In 2022, the basic EPS of $10.10 versus a diluted EPS of $9.95 reflect_
