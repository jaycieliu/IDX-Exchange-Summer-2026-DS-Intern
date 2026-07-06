# Week 3 Summary - Data Preprocessing

## Overview

Week 3 focused on turning the explored CRMLS sold-property dataset into a cleaned and model-ready dataset for California residential close price prediction. The work combined missing-value handling, categorical encoding, numerical preprocessing, and a time-based train/test split into one reproducible notebook.

The target analysis population remained restricted to:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

## What Was Done

- Combined the team's preprocessing work into one final notebook: `02_preprocessing.ipynb`.
- Loaded 17 monthly CRMLS sold files covering `2025-01` through `2026-05`.
- Added a `source_month` field to track which monthly file each row came from.
- Applied the required Residential + SingleFamilyResidence filter.
- Converted date fields such as `CloseDate`, `ContractStatusChangeDate`, `PurchaseContractDate`, and `ListingContractDate` to datetime format.
- Removed rows that could not support modeling because of missing `ClosePrice` or missing `CloseDate`.
- Removed duplicate listings using `ListingKey`, keeping the latest available record.
- Created missing-value and numerical summary files for documentation.
- Converted binary Y/N fields into numeric `0/1` values.
- Added numerical quality flags for invalid or unusual property records.
- Removed strict numerical errors while keeping review-level outliers with flags.
- Removed rows with missing latitude or longitude because geographic analysis is required.
- Encoded categorical fields into numeric model-ready features.
- Created a time-based train/test split using the most recent close month as the test set.
- Exported quality-cleaned and model-ready CSV files for later modeling.

## Key Outputs

- Raw data loaded: 352,354 rows.
- Filtered Residential + SingleFamilyResidence rows: 175,188 rows.
- Duplicate `ListingKey` rows removed: 119 rows.
- Rows removed by strict numerical rules: 143 rows.
- Rows removed for missing coordinates: 17 rows.
- Final quality-cleaned dataset: 174,909 rows.
- Model-ready training set: 123,469 rows.
- Model-ready test set: 12,012 rows.
- Test month: `2026-05`.
- Training window: `2025-05` through `2026-04`.

## Data Quality Decisions

- `ClosePrice` and `CloseDate` were treated as required fields because they are necessary for supervised price prediction and time-based splitting.
- Duplicate listings were handled with `ListingKey` to avoid repeated transactions in the modeling dataset.
- Missing latitude and longitude values were removed instead of imputed because fake coordinates would damage geographic analysis.
- Binary listing attributes such as `ViewYN`, `PoolPrivateYN`, `AttachedGarageYN`, and `NewConstructionYN` were converted to `0/1`.
- Strict numerical errors were removed, including invalid coordinates, non-positive close prices, invalid year built values, and negative days on market.
- Review-level outliers were kept with flags because expensive, rural, or unusual single-family properties may still be valid.
- `FireplacesTotal` was excluded from model-ready numeric scaling because it was fully missing in the training data.

## Feature Preparation

- Numerical features were imputed using medians calculated from the training set only.
- Missingness flags were added for numerical fields before imputation.
- Low-cardinality categorical fields such as `Levels`, `AssociationFeeFrequency`, and `StateOrProvince` were one-hot encoded.
- High-cardinality geographic fields such as `City`, `PostalCode`, `CountyOrParish`, `MLSAreaMajor`, and school district fields were frequency encoded.
- Numerical scaled versions were created using training-set means and standard deviations.
- Train-only preprocessing was used to avoid leakage from the May 2026 test month.

## Train/Test Split

The split was time-based rather than random. This better matches the real business use case, where a model is trained on past closed sales and evaluated on the newest available market month.

- Test set: `2026-05`
- Training set: `2025-05` through `2026-04`
- Default training window: `X = 12` months

Candidate training windows were also summarized for `X = 3, 6, 9, 12, 15, 16`. The 12-month window was selected as the default because it balances recent market relevance with enough training data for model stability.

## Risks / Limitations

- The best value of `X` has not been proven yet; it should be evaluated during modeling with metrics such as R2, MAPE, and MdAPE.
- Review-level outliers remain in the dataset and should be analyzed later by price segment and error distribution.
- Frequency encoding captures category prevalence but does not fully represent local market quality or school effects.
- Removing rows with missing coordinates may exclude records with weaker geocoding quality.
- The preprocessing step prepares data for modeling but does not evaluate prediction performance yet.

## Next Steps (Week 4)

- Train a baseline linear regression model.
- Train and compare tree-based models such as Random Forest, XGBoost, or LightGBM.
- Evaluate models using R2, MAPE, and MdAPE.
- Compare model performance by price segment.
- Analyze error distribution and overfitting risk.
- Use the results to decide whether the 12-month training window should be adjusted.

