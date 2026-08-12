# Week 8 - Evaluation Expansion Summary

## Final Read

**Use the Week 7 XGBoost model for pricing review, not automatic pricing.** On the comparable June in-range population, it keeps strong overall fit and acceptable typical percentage error.

## Main Metrics

Main population: **June in-range** transactions only. Full June is a robustness check.

| Population | Rows | R2 | MAPE | MdAPE |
|---|---:|---:|---:|---:|
| June in-range | 12,566 | **0.911** | **12.17%** | **8.46%** |
| Full June robustness | 12,851 | 0.678 | 13.81% | 8.67% |

## Price-Band Insight

| Price band | Rows | Median price | MAPE | MdAPE | P90 APE |
|---|---:|---:|---:|---:|---:|
| Q1 lowest | 2,483 | $451,500 | 14.33% | 8.95% | 33.92% |
| Q2 | 2,354 | $688,944 | **10.30%** | **6.79%** | 24.11% |
| Q3 | 2,434 | $900,000 | 10.64% | 7.49% | **22.79%** |
| Q4 | 2,581 | $1,285,000 | 11.94% | 8.80% | 25.08% |
| Q5 highest | 2,714 | $2,220,000 | 13.39% | 10.88% | 28.06% |

## Interpretation

1. **Middle-price homes are the most reliable pricing-review segment.** Q2 and Q3 have the lowest typical percentage errors.
2. **The lowest band needs caution.** Q1 has the highest MAPE and P90 APE, so small dollar misses can become large percentage misses.
3. **The highest band still needs manual review.** Q5 has the highest MdAPE and large dollar error, consistent with luxury or unusual properties being harder to price.

## Decision Implication

Use the model to flag pricing reasonableness for typical homes, especially Q2-Q4. Require manual review for low-price outliers, luxury homes, and any property with unusual characteristics.

## Limitation

This evaluation uses one June test month. Segment results measure model reliability, not causal market behavior.
