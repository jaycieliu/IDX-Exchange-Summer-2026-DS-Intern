# California Residential Close Price Prediction

## Abstract

Use the final XGBoost model for pricing review and valuation triage, not automatic pricing. On the primary June 2026 test set, the model achieved:

| Population | Rows | R2 | MAPE | MdAPE |
|---|---:|---:|---:|---:|
| June test set | 12,566 | 0.911 | 12.17% | 8.46% |
| Full June robustness | 12,851 | 0.678 | 13.81% | 8.67% |

**Decision implication:** 
    The model is useful for flagging pricing reasonableness on typical homes, especially middle-price segments. Low-price outliers, luxury homes, and unusual properties still require manual comparable-sale review.

## Business Objective

Predict `ClosePrice` for California CRMLS sold single-family residential properties so real estate teams can support:

- pricing review before or after close,
- listing strategy and valuation QA,
- geographic and price-segment demand allocation,
- manual-review prioritization for high-risk properties.

Every modeling step is evaluated against decision usefulness, not only technical score improvement.

## Dataset Source

Primary source: CRMLS sold property exports stored locally under:

- `data/raw data/CRMLSSoldYYYYMM.csv`
- `data/california/CRMLSSoldYYYYMM.csv`

External enrichment:

- `data/external/california_school_district_areas_2025_26.geojson`

Scope restriction:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`
- California sold records only
- target variable: `ClosePrice`

The final cleaned modeling dataset is:

- `outputs/week3_preprocessing/crmls_sfr_quality_cleaned_202501_202606.csv`

## Preprocessing

The preprocessing workflow is implemented in `notebooks/02_preprocessing.ipynb` and supported by `scripts/build_week3_preprocessing_deliverable.py`.

Main steps:

1. Load and combine monthly CRMLS sold files.
2. Filter to Residential / SingleFamilyResidence records.
3. Parse dates and create `close_month`.
4. Validate `ClosePrice` and key property fields.
5. Handle missing values with train-fitted logic instead of blind row removal.
6. Create quality flags for unusual or invalid values.
7. Add school-district enrichment.
8. Build chronological train, validation, and test datasets.

Final modeling window:

| Split | Period | Purpose |
|---|---|---|
| Train | 2025-02 to 2026-04 | Fit preprocessing and models |
| Validation | 2026-05 | Select model / hyperparameters |
| Test | 2026-06 | Final locked evaluation |

Important rule: imputation, scaling, frequency encoding, feature selection, and outlier bounds are fit on training data only, then applied unchanged to validation and test.

## Features

The final advanced-model feature set keeps interpretable price drivers:

- property size: living area, lot size, bedroom and bathroom counts,
- property structure: age, stories, garage and parking fields,
- ratios: bed/bath ratio and living-area-to-lot ratio,
- geography: city, ZIP, county, latitude, longitude,
- school-district context: district name, county, locale, enrollment, area, density,
- amenities: pool, view, fireplace, attached garage, new construction,
- seasonality: close-month sine and cosine.

Target-derived and sale-process leakage fields are excluded from modeling. Examples include `ClosePrice`, direct close-price ratios, and fields that would not be available at listing-time prediction.

## Modeling Approach

The project started with an interpretable linear baseline, then tested tree-based models and gradient-boosted models. Selection used chronological validation before final testing to reduce look-ahead bias.

| Modeling step | Models | Selection basis | Best result |
|---|---|---|---|
| Linear baseline | Linear Regression | chronological test metrics | R2 0.641, MAPE 34.60%, MdAPE 25.13% |
| Tree-based comparison | Linear Regression, Decision Tree, Random Forest | validation MdAPE, then MAPE, then R2 | Random Forest, test R2 0.878, MAPE 12.82%, MdAPE 8.32% |
| Feature-engineered pipeline comparison | 13 full pipelines using fixed and engineered feature sets | May validation before June test | `X5_fixed + Random Forest`, June test R2 0.877, MAPE 12.90%, MdAPE 8.29% |
| Advanced model comparison | XGBoost, LightGBM, CatBoost, Random Forest benchmark | balanced validation performance and train-validation stability | XGBoost, June test R2 0.911, MAPE 12.17%, MdAPE 8.46% |

Final selected model:

- model: XGBoost
- max depth: 6
- learning rate: 0.05
- effective trees: 1,495
- row sample: 0.85
- feature sample: 0.85
- L2 regularization: 1.0

Random Forest had slightly lower June MdAPE than XGBoost, but XGBoost had stronger R2, lower dollar error, and a smaller train-to-validation deterioration pattern. That makes XGBoost the better final model for pricing-review balance.

## Segment Performance

June test-set price-band diagnostics:

| Price band | Rows | Median price | MAPE | MdAPE | P90 APE |
|---|---:|---:|---:|---:|---:|
| Q1 lowest | 2,483 | $451,500 | 14.33% | 8.95% | 33.92% |
| Q2 | 2,354 | $688,944 | 10.30% | 6.79% | 24.11% |
| Q3 | 2,434 | $900,000 | 10.64% | 7.49% | 22.79% |
| Q4 | 2,581 | $1,285,000 | 11.94% | 8.80% | 25.08% |
| Q5 highest | 2,714 | $2,220,000 | 13.39% | 10.88% | 28.06% |

Decision implication:

- Q2 and Q3 are the most reliable pricing-review segments.
- Q1 needs caution because small dollar misses create high percentage misses.
- Q5 requires manual review because luxury and unusual properties have larger dollar risk.

## Key Outputs

| Output | Path |
|---|---|
| Cleaned modeling dataset | `outputs/week3_preprocessing/crmls_sfr_quality_cleaned_202501_202606.csv` |
| Random Forest model artifact | `outputs/week5_model_comparison/week5_selected_model.joblib` |
| Locked Random Forest pipeline | `outputs/week6_feature_engineering/week6_locked_pipeline.joblib` |
| Deployment refit pipeline | `outputs/week6_feature_engineering/week6_deployment_pipeline_refit_through_june.joblib` |
| Final evaluation metrics | `outputs/week8_evaluation/metrics_summary.csv` |
| Final XGBoost June predictions | `outputs/week8_evaluation/week8_xgboost_june_predictions.csv` |


## Assumptions

- CRMLS sold data is the authoritative source for this internship project.
- Chronological validation is required because real estate data shifts over time.
- The primary June test set is the main benchmark used for model reporting.
- Full June is shown separately as a robustness check because it includes more unusual records and is not the main model-selection basis.
- Price-band diagnostics are retrospective because they use realized `ClosePrice`.

## Limitations

- Only one validation month and one final test month are used in the final advanced-model readout.
- Frequency encoding captures category prevalence, not true neighborhood price level.
- School-district enrichment needs additional geospatial QA before production use.
- Luxury, low-price, and unusual properties remain higher-risk segments.
- The model predicts sold close price, not a guaranteed listing-time value.
- Feature importance is model-behavior evidence, not causal evidence.

## Final Decision

The final model is XGBoost. It should be used as a decision-support tool for pricing review, valuation QA, and manual-review triage. It should not be used to set final listing prices without human review and comparable-sale analysis.
