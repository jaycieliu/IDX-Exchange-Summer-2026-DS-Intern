# California Residential Price Prediction

## Executive Summary

This project built a model to estimate the final sale price of California single-family homes using historical CRMLS sold-property data.

The model should be used as a pricing-review tool, not as an automatic pricing decision. It is strongest for typical middle-market homes and weaker for unusual, very low-price, or luxury properties.

**Main result:** on the primary June 2026 test set, the final model explained most of the price variation and had a typical prediction error of about **8.5%**.

| Evaluation view | Homes reviewed | Overall fit | Average error | Typical error |
|---|---:|---:|---:|---:|
| Main June test set | 12,566 | 0.911 | 12.17% | 8.46% |
| Full June robustness check | 12,851 | 0.678 | 13.81% | 8.67% |

Plain-English interpretation:

- **Overall fit 0.911** means the model captured most of the pricing pattern in the main test group.
- **Average error 12.17%** means some hard cases still create larger misses.
- **Typical error 8.46%** is the better summary for what a normal prediction miss looks like.
- The full June check performs worse because it includes more unusual properties. Those should be reviewed manually.

## Business Use

The model is useful for:

- checking whether a sale price looks reasonable,
- flagging homes that may need manual review,
- comparing pricing risk across price ranges,
- supporting agent or analyst valuation review.

The model is not appropriate for:

- setting final listing prices by itself,
- replacing comparable-sale analysis,
- making high-stakes pricing decisions without human review,
- pricing luxury or unusual homes without extra judgment.

## Data Used

The project uses California CRMLS sold-property records. Each record describes a closed sale, including property size, bedrooms, bathrooms, lot information, location, listing timing, and final close price.

Only this property type was included:

- Residential
- Single-family residence
- California sold transactions

School-district boundary data was added to give the model more location context. This helps the model understand that nearby properties can belong to different school-district areas, which may affect pricing behavior.

## How The Data Was Prepared

The raw monthly property files were cleaned and combined into one modeling dataset.

Main preparation steps:

1. Kept only California residential single-family sold homes.
2. Removed or flagged records with invalid price or property values.
3. Filled missing values using rules learned from the training data.
4. Converted location and category fields into model-ready values.
5. Added school-district information using property location.
6. Split the data by time so the model was tested on future months, not random rows.

The time-based split is important. Real estate markets change over time, so the model must prove that it works on later sales, not just on homes from the same period it learned from.

## What The Model Looks At

The model uses practical pricing signals:

- home size,
- lot size,
- bedrooms and bathrooms,
- property age,
- parking and garage information,
- location,
- ZIP code, city, county, and school-district context,
- amenities such as pool, view, fireplace, and new construction,
- seasonality based on closing month.

Fields that could leak the answer were excluded. For example, the model should not rely on values that directly reveal or are derived from the final close price.

## Models Compared

The project compared simple and more advanced models to make sure the final result was truly better than a basic benchmark.

| Model type | What it showed | Result |
|---|---|---|
| Linear regression | Simple baseline; useful for comparison but not accurate enough | Typical error about 25% |
| Decision tree / Random Forest | Much stronger pricing pattern detection | Random Forest typical error about 8.3% |
| Gradient boosting models | Best overall balance of accuracy and stability | XGBoost selected as final model |

The final model was selected because it had the best balance of:

- strong overall fit,
- lower dollar error,
- acceptable typical percentage error,
- better stability from validation to final testing.

## Where The Model Works Best

The model performs best on middle-price homes. These are the homes where it is most useful for pricing review.

| Price range group | Homes reviewed | Median sale price | Average error | Typical error |
|---|---:|---:|---:|---:|
| Lowest-price group | 2,483 | $451,500 | 14.33% | 8.95% |
| Lower-middle group | 2,354 | $688,944 | 10.30% | 6.79% |
| Middle group | 2,434 | $900,000 | 10.64% | 7.49% |
| Upper-middle group | 2,581 | $1,285,000 | 11.94% | 8.80% |
| Highest-price group | 2,714 | $2,220,000 | 13.39% | 10.88% |

Key takeaways:

1. **Middle-price homes are the best use case.** The lower-middle and middle groups have the lowest typical errors.
2. **Lowest-price homes need caution.** A smaller dollar miss can become a large percentage miss.
3. **Highest-price homes need manual review.** Luxury and unusual homes are harder to estimate reliably.

## Recommendation

Use the model as a first-pass pricing review system:

1. Score a property or group of properties.
2. Flag homes where the predicted price and actual/listed price are far apart.
3. Prioritize flagged homes for analyst or agent review.
4. Use comparable sales and local market judgment before making a final pricing decision.

This is the right level of use because the model is accurate enough to support review, but not reliable enough to replace expert pricing judgment.

## Main Limitations

- The model was evaluated on one final test month, so additional future-month testing is needed.
- Location is represented through available fields and school-district context, but it does not fully replace neighborhood expertise.
- Luxury, unusual, and very low-price properties have higher risk.
- The model predicts historical final sale price, not guaranteed listing price.
- School-district enrichment is useful, but should receive additional quality checks before production use.

## App Status

A Streamlit prediction app is not currently included in this repository.

If an app is added, it should let a user enter basic property details, return a predicted price, and show a short explanation of the main drivers. It should also clearly state that the output is for pricing review only.

Expected launch command after an app exists:

```bash
streamlit run app.py
```


## Final Decision

The final XGBoost model is suitable for pricing review, valuation quality checks, and manual-review prioritization. It should not be used as a standalone pricing engine.
