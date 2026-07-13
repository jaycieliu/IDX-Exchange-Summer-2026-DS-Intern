# Week 4 Summary - Baseline Model

## Overview

Week 4 trained and evaluated baseline Linear Regression models for California residential single-family ClosePrice prediction. The purpose was to test several non-leaky independent-variable bundles and select the strongest baseline feature set before moving to more advanced models.

The target analysis population remained restricted to:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

The modeling target was:

- `Y = ClosePrice`

## Business Objective

The baseline model supports real estate valuation and pricing decisions by estimating expected close price from property characteristics available for both on-market and off-market homes.

The Week 4 decision question was:

> Which non-leaky feature bundle gives the strongest baseline Linear Regression performance for predicting May 2026 ClosePrice?

## What Was Done

- Used the Week 3 model-ready train/test split.
- Kept the standardized chronological split:
  - Training window: `2025-05` through `2026-04`
  - Test month: `2026-05`
- Confirmed that `ClosePrice` is the prediction target.
- Excluded target leakage features from all models:
  - `ListPrice`
  - `OriginalListPrice`
  - `ClosePrice_to_ListPrice_ratio`
- Started with one key independent variable: `LivingArea_scaled`.
- Checked training-set correlation between `LivingArea_scaled` and `ClosePrice`.
- Screened candidate features using correlation with `ClosePrice`.
- Checked multicollinearity among candidate X variables.
- Trained OLS Linear Regression models on the training data.
- Evaluated each feature bundle on the May 2026 test set.
- Compared models using:
  - R2
  - MAPE
  - MdAPE

## Feature Screening Findings

- `LivingArea_scaled` had a training correlation of `0.6192` with `ClosePrice`.
- This made living area a defensible first independent variable, but not sufficient by itself.
- `ElementarySchoolDistrict_frequency` and `MiddleOrJuniorSchoolDistrict_frequency` were removed because the raw school district fields were entirely `Unknown` in the training set.
- High-correlation X-to-X pairs included:
  - `Latitude_scaled` and `Longitude_scaled`
  - `LivingArea_scaled` and `BathroomsTotalInteger_scaled`
  - `Latitude_scaled` and `MLSAreaMajor_frequency`

## Multicollinearity Decisions

- `Latitude_scaled` and `Longitude_scaled` were retained only in location bundles because together they represent geographic position, but they are rough linear location proxies.
- `LivingArea_scaled` was retained as the core size variable.
- `BathroomsTotalInteger_scaled` was tested separately because it is correlated with living area but may still add useful signal.
- `MLSAreaMajor_frequency` was excluded from the first location bundle because it overlapped with latitude and was only a frequency-encoded proxy.
- `HighSchoolDistrict_frequency` was tested in the expanded bundle, but it was interpreted only as a rough frequency encoding, not school quality.

## Model Bundles

### Model 1: Size Only

- `LivingArea_scaled`

### Model 2: Basic Structure

- `LivingArea_scaled`
- `BedroomsTotal_scaled`
- `YearBuilt_scaled`
- `LotSizeSquareFeet_scaled`

### Model 3: Structure + Bathrooms

- Model 2 features
- `BathroomsTotalInteger_scaled`

### Model 4: Structure + Location

- Model 3 features
- `Latitude_scaled`
- `Longitude_scaled`
- `PostalCode_frequency`
- `City_frequency`

### Model 5: Expanded Non-Leaky Bundle

- Model 4 features
- `AssociationFee_scaled`
- `Stories_scaled`
- `GarageSpaces_scaled`
- `DaysOnMarket_scaled`
- `HighSchoolDistrict_frequency`

## Training Results

| Model | Train R2 |
|---|---:|
| Model 1: Size Only | 0.383 |
| Model 2: Basic Structure | 0.440 |
| Model 3: Structure + Bathrooms | 0.457 |
| Model 4: Structure + Location | 0.495 |
| Model 5: Expanded Non-Leaky Bundle | 0.506 |

Training R2 increased as more feature groups were added. The largest conceptual improvement came from adding location features in Model 4.

## Test Results

| Model | Train R2 | Test R2 | Test MAPE | Test MdAPE |
|---|---:|---:|---:|---:|
| Model 5: Expanded Non-Leaky Bundle | 0.506 | 0.537 | 0.496 | 0.330 |
| Model 4: Structure + Location | 0.495 | 0.528 | 0.508 | 0.339 |
| Model 3: Structure + Bathrooms | 0.457 | 0.479 | 0.533 | 0.362 |
| Model 2: Basic Structure | 0.440 | 0.465 | 0.538 | 0.356 |
| Model 1: Size Only | 0.383 | 0.402 | 0.601 | 0.418 |

## Selected Baseline Bundle

The selected Week 4 baseline bundle is:

> Model 5: Expanded Non-Leaky Bundle

Model 5 was selected because it had:

- Highest test R2: `0.537`
- Lowest test MAPE: `0.496`
- Lowest test MdAPE: `0.330`

## Decision Implication

The results show that home size alone is not enough for reliable price prediction. Physical structure improves performance, location improves performance more meaningfully, and the expanded non-leaky feature bundle performs best overall.

For pricing and valuation decisions, the baseline confirms that property size, structure, location, and selected market/context variables all contribute to ClosePrice prediction. However, the baseline error remains high, so this Linear Regression model should be treated as a benchmark rather than a final valuation model.

## Assumptions

- The Week 3 preprocessing outputs are the source of truth for modeling.
- The May 2026 test set is the latest full available close month.
- All feature screening was based on the training set only.
- Frequency-encoded features represent category prevalence, not quality or price level.
- Linear latitude and longitude are rough geographic proxies.

## Limitations

- Test MdAPE around `33%` is too high for production valuation use.
- OLS coefficients are not causal effects.
- P-values are easy to make significant because the training set is large.
- Some location relationships are nonlinear and not well captured by Linear Regression.
- `HighSchoolDistrict_frequency` is not a school-quality measure.
- School district enrichment should be revisited in Week 6 with a proper geographic layer.

