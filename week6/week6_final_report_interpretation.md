## Week 6 Final Report Interpretation

### Executive Conclusion

- Validation selected `X5_fixed + Random Forest` as the locked Week 6 pipeline.
- Week 6 keeps the same core benchmark structure as Week 5.
- The updated raw-to-prediction pipeline now rebuilds preprocessing and engineered features consistently.
- Primary June testing uses all eligible June SFR transactions.
- Primary June performance: R2 `0.5950`, MAPE `14.95%`, MdAPE `8.55%`.
- Secondary in-range June performance: R2 `0.8770`, MAPE `12.90%`, MdAPE `8.29%`.
- Week 6 reproduces the Week 5 RF benchmark under a cleaner pipeline, but does not show a meaningful model improvement.

### Validation Decision

- May validation is used only to choose among completed pipelines.
- Feature engineering, preprocessing, feature selection, and model fitting are completed using training data only.
- June is not used during pipeline selection.
- `X5_fixed + Random Forest` wins because it has the strongest validation MdAPE.
- X6 engineered and selected pipelines were tested, but did not justify replacing the X5 RF benchmark.
- The locked pipeline uses `70` transformed features after removing zero-variance features.

### Controlled Feature-Set Findings

- **School layer:** improves X5 RF validation MdAPE by `0.18 percentage points`.
- **School layer:** improves X6 RF validation MdAPE by `0.13 percentage points`.
- **Engineered features:** X6 full does not outperform X5 full; MdAPE worsens by `0.06 percentage points`.
- **Feature selection:** RF selection reduces feature count, but MdAPE improves by only `0.02 percentage points`.
- **Decision:** keep `X5_fixed + Random Forest` as the locked Week 6 model.

### Why Week 5 And Week 6 Differ Slightly

- The benchmark specification is unchanged: both use `X5_fixed + Random Forest`.
- The RF hyperparameters and `random_state` are the same.
- Week 6 correctly retrains the model inside the updated pipeline.
- Retraining alone is not the main reason for metric differences.
- With identical transformed inputs, row order, feature order, parameters, and random seed, RF predictions should reproduce closely.
- The main difference comes from preprocessing: Week 6 rebuilds the transformed feature matrix with cleaner missing-value, unknown-category, binary, postal-code, and zero-variance handling.
- The current in-range June populations are identical: both evaluate `12,566` June transactions.
- Performance is effectively unchanged: Week 6 has slightly lower R2 but slightly better MdAPE.
- Correct conclusion: the small mixed metric differences reflect preprocessing implementation changes, not evidence of a meaningfully better model.

### June Test Interpretation

- Primary June testing is stricter because it includes all eligible June SFR sales.
- The lower primary R2 shows that extreme June transactions strongly affect squared-error performance.
- The secondary in-range check shows much stronger performance on typical homes.
- This means the model is more stable for homes similar to the training population.
- Extreme, unusual, or high-risk transactions still require manual review.

### Segment Risk

- Q2 and Q3 homes are the most stable prediction segments.
- Q1 low-price homes have high tail risk, with MdAPE `8.61%` and P90 APE `49.58%`.
- Q5 high-price homes have higher absolute error and weaker percentage accuracy, with MdAPE `12.60%` and P90 APE `36.66%`.
- Manual review is still required for low-price outliers, luxury homes, abnormal price-per-sqft records, and weak location matches.

### Business Decision

- Listing-price review: use the model as directional pricing support.
- Valuation triage: use prediction errors and segment risk to prioritize manual review.
- Automated pricing: not recommended because tail risk remains material.
- Week 7 should focus on rolling-origin validation, prediction-time feature availability, and segment-level reliability.
