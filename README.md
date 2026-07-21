# IDX-Exchange-DS-Intern: California Property Close Price Prediction

## Project Overview

This repository contains the internship project for predicting California residential property close prices using historical CRMLS sold property data.

The goal of this project is to build a machine learning workflow that can predict a property's final sales price, also known as `ClosePrice`, based on property-level characteristics such as living area, bedrooms, bathrooms, lot size, location-related fields, and other available listing attributes.

For this project, the analysis focuses only on:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`
- California CRMLS sold property records

The final deliverables include data preprocessing scripts, model training and evaluation code, documentation, and a final presentation summarizing the methodology, findings, model performance, and next steps.

---

## Week 1 Summary: Orientation & Setup

**Week 1 deliverables completed:**

- Reviewed the task prompt and project objective.
- Set up Python, Git, Jupyter Notebook, and the local project environment.
- Connected to the FTP server through FileZilla.
- Located and downloaded CRMLS sold property files.
- Reviewed the metadata file to understand key feature definitions.
- Confirmed dataset access and prepared initial notes on important columns.


## Week 2 Summary: Data Exploration

**Week 2 deliverables completed:**

- Loaded at least six months of CRMLS sold data into pandas.
- Combined multiple monthly files into one working dataset.
- Filtered the data to residential single-family properties.
- Explored the distribution of `ClosePrice`, living area, bedrooms, bathrooms, and lot size.
- Checked missing values and basic data quality issues.
  
**Documentation:**
  - [Week2 Summary](/week2/week2.md)
    
**Codes:**
  - [Week2 Notebook](/week2/01_exploration.ipynb)
    
## Week 3 Summary: Data Preprocessing

**Week 3 deliverables completed:**

- Combined the team's preprocessing work into one final notebook.
- Loaded and combined CRMLS sold files from January 2025 through May 2026.
- Filtered the dataset to residential single-family properties.
- Handled missing values, duplicates, binary fields, categorical encoding, and numerical quality issues.
- Added numerical flags for invalid or unusual property records.
- Created a time-based train/test split:
  - Train: `2025-05` to `2026-04`
  - Test: `2026-05`
- Exported quality-cleaned and model-ready CSV files for future modeling.

**Final preprocessing output:**

- Quality-cleaned rows: `181,185`
- Training rows: `129,745`
- Test rows: `12,012`

**Documentation:**
- [Week3 Summary](/week3/week3.md)

**Codes:**
- [Week3 Notebook](/week3/02_preprocessing.ipynb)


## Week 4 Summary: Baseline Model

**Week 4 deliverables completed:**

- Built the first baseline Linear Regression model for predicting `ClosePrice`.
- Used the Week 3 chronological train/test split:
  - Train: `2025-05` to `2026-04`
  - Test: `2026-05`
- Excluded leakage features:
  - `ListPrice`
  - `OriginalListPrice`
  - `ClosePrice_to_ListPrice_ratio`
- Checked feature correlation and multicollinearity before modeling.
- Tested multiple non-leaky X feature bundles.
- Compared models using:
  - R²
  - MAPE
  - MdAPE
- Selected the best baseline X bundle based on test-set performance.

**Best baseline model:**

- Selected model: `Model 5 - Expanded Non-Leaky Bundle`
- Test R²: `0.537`
- Test MAPE: `0.496`
- Test MdAPE: `0.330`

**Decision:**

- Model 5 was selected as the Week 4 baseline because it had the strongest test-set R² and lowest error metrics.
- The Linear Regression baseline is useful as a benchmark, but the error level is still too high for final pricing decisions.

**Documentation:**
- [Week4 Summary](/week4/week4.md)

**Codes:**
- [Week4 Notebook](/notebooks/03_baseline_model.ipynb)

## Week 5 Summary: Additional Models

**Week 5 deliverables completed:**

- Revised Week 2-4 documentation and methodology summaries.
- Updated Week 3 preprocessing with school-district enrichment and corrected validation/test split.
- Updated Week 4 Linear Regression baseline to use validation for model selection.
- Built additional regression models for predicting `ClosePrice`.
- Used the same locked modeling setup as Week 4:
  - Train: `2025-04` to `2026-03`
  - Validation: `2026-04`
  - Test: `2026-05`
- Reused the Week 4 `X5_full_non_leaky` feature set.
- Compared three model families:
  - Linear Regression baseline
  - Decision Tree Regressor
  - Random Forest Regressor
- Tuned tree-model hyperparameters using validation data only.
- Selected the final model based on validation performance:
  - R²
  - MAPE
  - MdAPE
- Tested the locked final model once on the May 2026 test set.
- Analyzed model behavior using:
  - price-segment errors
  - error distribution
  - feature importance
  - actual vs predicted plot

**Best additional model:**

- Selected model: `Random Forest`
- Parameters: `n_estimators=50, max_depth=22, min_samples_leaf=10, max_features=0.7`
- Test R²: `0.872`
- Test MAPE: `0.131`
- Test MdAPE: `0.085`

**Decision:**

- Random Forest was selected because it had the strongest validation MdAPE and materially improved test performance compared with the Week 4 Linear Regression baseline.
- The model is useful for directional pricing support and valuation triage.
- The model should not be used for automated final pricing because segment-level and tail errors remain meaningful.

**Documentation:**

- [Week 5 Summary](week5/week5.md)

**Code:**

- [Week 5 Notebook](notebooks/04_model_comparison.ipynb)


- # Week 3-4 Revision Summary
## Week 3 - Preprocessing

- Rebuilt preprocessing from raw CRMLS data instead of combining teammate code blindly.
- Documented filtering, duplicate handling, missing-value treatment, and outlier flags.
- Added train-only imputation, scaling, frequency encoding, and missingness indicators.
- Added official Unified School District relative/geographic variables.
- Set validation to `2026-04` and final test to `2026-05`.

## Week 4 - Linear Baseline

- Refactored baseline modeling into train, validation, and test phases.
- Used validation, not test, to select the training window and X set.
- Compared five X sets across five training windows with `R2`, `MAPE`, and `MdAPE`.
- Selected `X5_full_non_leaky` with the 12-month window as the locked Linear Regression baseline.
- Reserved May 2026 test for final one-time baseline evaluation.

