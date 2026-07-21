# Week 4 Updated Summary - Linear Regression Baseline

## Objective

Week 4 establishes the Linear Regression baseline for California residential single-family `ClosePrice` prediction before Week 5 tree-based models.

The business decision is whether a transparent linear baseline is strong enough for pricing support, listing-price review, and valuation triage, or whether nonlinear models are needed.

Target population:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

Target:

- `ClosePrice`

## Concise Revision Summary For Weeks 2-4

- **Week 2:** tightened EDA around the required Residential + SingleFamilyResidence population, key price/size distributions, missingness, duplicate checks, and data-quality risks.
- **Week 3:** revised preprocessing from raw CRMLS data, kept train-only transformations, added school-district relative/geographic variables, and set validation to `2026-04` with test held out as `2026-05`.
- **Week 4:** corrected the baseline workflow so validation, not test, selects the final Linear Regression feature bundle and training window.

## Correct Train / Validation / Test Workflow

The notebook now uses a proper chronological ML evaluation workflow.

| Split | Month logic | Purpose |
| --- | --- | --- |
| Train | rolling windows ending `2026-03` | Fit preprocessing and Linear Regression coefficients |
| Validation | `2026-04` | Select the training window and X-set |
| Test | `2026-05` | Final evaluation after the baseline is locked |

No random `train_test_split` is used because this is a real estate forecasting problem with temporal dependency.

## Candidate Training Windows

Week 4 compares five historical training windows:

- 3 months
- 6 months
- 9 months
- 12 months
- 15 months

Each window is evaluated on April 2026 validation. May 2026 test is not used during model selection.

## Candidate X Sets

The notebook compares five feature bundles:

| X set | Description |
| --- | --- |
| `X1_size` | Living area only |
| `X2_physical` | Size, bedrooms, bathrooms, lot size, age, and physical ratios |
| `X3_location` | Physical features plus latitude, longitude, city, ZIP, county, and MLS area |
| `X4_school_location` | Location features plus official Unified School District variables |
| `X5_full_non_leaky` | Physical, location, school district, amenity, and seasonality variables |

`X5_full_non_leaky` has `77` transformed features after train-fitted preprocessing.

## Leakage Controls

The notebook excludes leakage-prone fields from candidate X sets, including:

- `ClosePrice`
- `ListPrice`
- `OriginalListPrice`
- `ClosePrice_to_ListPrice_ratio`
- identifiers
- agent and office fields
- post-close date fields

Categorical variables use frequency encoding, not target encoding. That means category prevalence is used, not average price by category.

## Preprocessing Method

All preprocessing is fit on training data only:

- numeric variables: train-median imputation, scaling, and missingness flags
- categorical variables: train-frequency encoding plus unknown flags
- binary variables: train-mode fill
- outlier filters: `ClosePrice` and `price_per_sqft_audit` cutoffs learned from train only

The same frozen preprocessing is then applied to validation and test.

## Validation Results

Top validation results:

| Training window | X set | Features | Validation R2 | Validation MAPE | Validation MdAPE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12 | `X5_full_non_leaky` | 77 | 0.6390 | 0.3521 | 0.2501 |
| 9 | `X5_full_non_leaky` | 77 | 0.6410 | 0.3514 | 0.2502 |
| 3 | `X5_full_non_leaky` | 77 | 0.6283 | 0.3527 | 0.2509 |
| 15 | `X5_full_non_leaky` | 77 | 0.6374 | 0.3525 | 0.2511 |
| 6 | `X5_full_non_leaky` | 77 | 0.6396 | 0.3568 | 0.2525 |

Selection rule:

1. lowest validation `MdAPE`
2. lowest validation `MAPE`
3. highest validation `R2`

## Selected Baseline

The selected Week 4 baseline is:

- Model type: `LinearRegression`
- Training window: `2025-04` to `2026-03`
- Validation month: `2026-04`
- Test month: `2026-05`
- X set: `X5_full_non_leaky`
- Transformed feature count: `77`

## Final Metrics

| Split | R2 | MAPE | MdAPE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.6322 | 0.3536 | 0.2537 | 360420.9623 |
| Validation | 0.6390 | 0.3521 | 0.2501 | 373716.6617 |
| Test | 0.6338 | 0.3570 | 0.2551 | 374446.3837 |

## Result Interpretation

The validation and test metrics are close, so the baseline does not show a major generalization collapse. However, test MdAPE is still about `25.5%`, which is too high for automated valuation.

The model is useful as a benchmark, not as the final AVM. Week 5 models must beat this locked baseline on validation before being evaluated on the May 2026 test month.

## Price-Segment Error

May 2026 test performance by price segment:

| Price segment | Rows | Median close price | MAPE | MdAPE |
| --- | ---: | ---: | ---: | ---: |
| Q1 lowest | 2362 | 455000 | 0.5848 | 0.4347 |
| Q2 | 2341 | 700000 | 0.3987 | 0.2901 |
| Q3 | 2352 | 929444 | 0.3146 | 0.2342 |
| Q4 | 2362 | 1343250 | 0.2294 | 0.1761 |
| Q5 highest | 2341 | 2307500 | 0.2567 | 0.2376 |

The baseline is weakest in percentage-error terms for lower-price homes and carries large dollar-error risk for expensive homes.

## Decision Implication

The linear baseline confirms that price prediction needs more than size. The best baseline combines physical structure, geography, school-district segmentation, amenities, and seasonality.

For business use, this baseline can support model benchmarking and explain the value of broader non-leaky features. It should not be used as a final pricing model because typical test error remains too high.

## Assumptions

- `CloseDate` defines the chronological split.
- April 2026 is the validation month.
- May 2026 is the final test month.
- Frequency encoding represents market prevalence, not category quality.
- Unified School District features are geographic segmentation features, not causal school-quality measures.

## Limitations

- This is Linear Regression, so nonlinear location and property interactions are under-modeled.
- One validation month and one test month are not enough for production approval.
- Coefficients are not causal effects.
- Missing condition, renovation quality, view quality, and local competition still limit pricing precision.
- Rolling-origin backtesting is required before any production AVM claim.
