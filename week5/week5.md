# Week 5 Summary - Additional Models

## What I Completed

- Reused the Week 4 modeling setup: same `2025-04` to `2026-03` training window, `2026-04` validation month, `2026-05` test month, train-only outlier filters, and `77` transformed `X5_full_non_leaky` variables.
- Trained the fixed Week 4 Linear Regression baseline plus Decision Tree and Random Forest candidates.
- Tried a small readable hyperparameter grid for tree models.
- Selected the final model using validation metrics only.
- Scored the locked selected model on the May 2026 test set once.
- Added price-segment error analysis, error-distribution analysis, feature-importance review, and inline `plt.show()` visualizations in `notebooks/04_model_comparison.ipynb`.

## Concise Revision Summary For Weeks 2-4

- **Week 2:** clarified the EDA scope around the required CRMLS population, core price/size distributions, missingness, duplicates, and data-quality risks that affect pricing decisions.
- **Week 3:** rebuilt preprocessing around the raw CRMLS data, added train-only cleaning and encoding logic, added official Unified School District relative/geographic variables, and fixed the split to validation `2026-04` and test `2026-05`.
- **Week 4:** refactored the Linear Regression baseline so training fits coefficients, validation selects the X set and training window, and May 2026 test is used only once after the baseline is locked.

## Objective

Week 5 compares additional models for California residential single-family `ClosePrice` prediction. The business decision is whether a tuned tree-based model should replace the Week 4 Linear Regression baseline for pricing, listing strategy, and demand allocation support.

Target population:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

Target:

- `ClosePrice`

## Week 4 Baseline Alignment

Week 4 selected:

- Training window: `2025-04` to `2026-03` (`12` months)
- Validation month: `2026-04`
- Test month: `2026-05`
- X set: `X5_full_non_leaky`
- Transformed feature count: `77`

Week 5 uses the same source file, split, train-only outlier filters, and transformed X variables before comparing model families.

## Correct Train / Validation / Test Logic

1. **Training phase:** fit each candidate on the training set only. For trees, training learns split rules and leaf predictions for each predefined hyperparameter setting.
2. **Validation phase:** choose the best hyperparameters and model family using validation `MdAPE`, then validation `MAPE`, then validation `R2`.
3. **Testing phase:** after the winner is locked, score May 2026 test once for the selected model and fixed Week 4 baseline.

The test month is not used to tune parameters or choose among candidates.

## Hyperparameter Search

The search is intentionally small and readable:

- Linear Regression: fixed Week 4 baseline
- Decision Tree: 4 depth/leaf settings
- Random Forest: 4 depth/leaf/max-feature settings with `n_estimators=50`

## Best Validation Result By Family

| model_family | parameters | train_R2 | validation_R2 | validation_MAPE | validation_MdAPE | train_minus_validation_R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | max_depth=20, min_samples_leaf=50 | 0.8286 | 0.8042 | 0.1669 | 0.1081 | 0.0244 |
| Linear Regression baseline | default | 0.6322 | 0.6390 | 0.3521 | 0.2501 | -0.0067 |
| Random Forest | n_estimators=50, max_depth=22, min_samples_leaf=10, max_features=0.7 | 0.9189 | 0.8728 | 0.1307 | 0.0832 | 0.0460 |

## Full Validation Search Results

| model_family | parameters | train_R2 | validation_R2 | validation_MAPE | validation_MdAPE | train_minus_validation_R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | n_estimators=50, max_depth=22, min_samples_leaf=10, max_features=0.7 | 0.9189 | 0.8728 | 0.1307 | 0.0832 | 0.0460 |
| Random Forest | n_estimators=50, max_depth=22, min_samples_leaf=10, max_features=0.6 | 0.9176 | 0.8716 | 0.1312 | 0.0837 | 0.0460 |
| Random Forest | n_estimators=50, max_depth=16, min_samples_leaf=20, max_features=0.6 | 0.8842 | 0.8563 | 0.1449 | 0.0934 | 0.0280 |
| Random Forest | n_estimators=50, max_depth=16, min_samples_leaf=20, max_features=0.7 | 0.8845 | 0.8558 | 0.1442 | 0.0937 | 0.0287 |
| Decision Tree | max_depth=20, min_samples_leaf=50 | 0.8286 | 0.8042 | 0.1669 | 0.1081 | 0.0244 |
| Decision Tree | max_depth=16, min_samples_leaf=50 | 0.8268 | 0.8032 | 0.1701 | 0.1113 | 0.0236 |
| Decision Tree | max_depth=12, min_samples_leaf=100 | 0.7763 | 0.7681 | 0.2020 | 0.1349 | 0.0082 |
| Decision Tree | max_depth=8, min_samples_leaf=100 | 0.7100 | 0.7066 | 0.2681 | 0.1961 | 0.0034 |
| Linear Regression baseline | default | 0.6322 | 0.6390 | 0.3521 | 0.2501 | -0.0067 |

## Selected Model Before Test

Selected model: `Random Forest`

Selected parameters: `n_estimators=50, max_depth=22, min_samples_leaf=10, max_features=0.7`

Validation selection metrics:

- Validation R2: `0.8728`
- Validation MAPE: `0.1307`
- Validation MdAPE: `0.0832`

## Final Test After Selection

| model | R2 | MAPE | MdAPE | MAE |
| --- | --- | --- | --- | --- |
| Selected model: Random Forest | 0.8720 | 0.1306 | 0.0846 | 173051.6842 |
| Week 4 baseline: Linear Regression | 0.6338 | 0.3570 | 0.2551 | 374446.3837 |

Compared with the Week 4 Linear Regression baseline, the selected Random Forest reduces test MdAPE by about `67.2%` and test MAPE by about `63.4%`.

## Visual Results

The notebook displays four figures inline with `plt.show()` only:

- Validation MdAPE by candidate
- Best validation MdAPE by model family
- Selected-model test error by price segment
- Actual vs predicted ClosePrice on the May 2026 test sample

No figure files are saved or linked from this summary.

## Price-Segment Test Performance

| price_segment | rows | median_close_price | mae | mape | mdape | p90_ape |
| --- | --- | --- | --- | --- | --- | --- |
| Q1_lowest | 2362 | 455000.0000 | 61653.1068 | 0.1483 | 0.0755 | 0.3662 |
| Q2 | 2341 | 700000.0000 | 76407.4485 | 0.1097 | 0.0627 | 0.2375 |
| Q3 | 2352 | 929444.0000 | 104950.1546 | 0.1112 | 0.0735 | 0.2480 |
| Q4 | 2362 | 1343250.0000 | 171304.3966 | 0.1268 | 0.0917 | 0.2546 |
| Q5_highest | 2341 | 2307500.0000 | 452278.2908 | 0.1568 | 0.1252 | 0.3226 |

## Error Distribution

| mean_error | median_error | p10_error | p90_error | p50_ape | p75_ape | p90_ape | p95_ape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -14631.3284 | 3480.9444 | -249500.1079 | 224516.3739 | 0.0846 | 0.1649 | 0.2885 | 0.3983 |

## Top Feature Importances

| feature | importance |
| --- | --- |
| LivingArea_scaled | 0.3566 |
| BathroomsTotalInteger_scaled | 0.1102 |
| Longitude_scaled | 0.0887 |
| Latitude_scaled | 0.0866 |
| PostalCode_frequency | 0.0565 |
| PoolPrivateYN_was_missing | 0.0391 |
| CountyOrParish_frequency | 0.0328 |
| MLSAreaMajor_frequency | 0.0292 |
| bed_bath_ratio_scaled | 0.0275 |
| UnifiedSchoolDistrictEnrollmentDensity_scaled | 0.0245 |

## Final Interpretation

### Decision

Use the tuned Random Forest as the Week 5 benchmark model. It is selected strictly from validation results. After selection, it beats the exact Week 4 Linear Regression baseline on the May 2026 test month using the same source data, split, train-only filters, and `77` transformed `X5_full_non_leaky` variables.

### Business Meaning

The selected Random Forest materially reduces typical pricing error. The Week 4 Linear Regression baseline has test MdAPE `0.2551`, while the selected Random Forest has test MdAPE `0.0846`. This makes the Random Forest more useful for directional valuation, listing-price review, and prioritizing properties for pricing attention.

The model is still not safe for automated final pricing. Segment results show that high-price homes have the largest dollar errors, and the error-distribution table shows meaningful tail risk: p90 APE is `0.2885` and p95 APE is `0.3983`.

### Hyperparameter Finding

The best Random Forest setting is selected from the validation grid only: `n_estimators=50, max_depth=22, min_samples_leaf=10, max_features=0.7`. The winning setting uses a deeper forest with smaller leaves than the conservative forest candidates, which suggests the data contains nonlinear interactions that Linear Regression misses. A single Decision Tree improves over Linear Regression but remains less accurate and less stable than the ensemble.

### Strengths

- Same Week 4 baseline setup, so the comparison is fair.
- Validation-based tuning avoids test-set selection leakage.
- Test set is scored only after the model and parameters are selected.
- Random Forest captures nonlinear structure missed by Linear Regression.
- MdAPE, MAPE, R2, segment errors, error distribution, feature importance, and inline plots are all reported.

### Limitations

- The grid search is intentionally small for readability; it is not exhaustive optimization.
- Only one validation month and one test month are used here.
- Feature importance is model behavior evidence, not causal proof.
- Missing interior condition, renovation quality, view quality, and active competition still limit pricing precision.
- Production approval requires rolling-origin backtesting across multiple historical cutoffs.

### Decision Implication

Use the tuned Random Forest as the Week 5 modeling benchmark and move it forward to the next feature-engineering comparison. Do not present it as a production AVM yet. The next decision should be whether additional engineered features reduce remaining segment-level error without increasing overfitting.
