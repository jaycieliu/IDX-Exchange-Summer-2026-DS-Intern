# Week 7 - Advanced Models

## Final Decision

**Select XGBoost as the final model.** It achieved the highest validation R2 and the lowest dollar error, while maintaining a much smaller train-to-validation error gap than Random Forest.

## 1. Plan

1. Reuse the cleaned data and feature logic from previous weeks.
2. Tune XGBoost, LightGBM, and CatBoost using May validation data.
3. Compare the best boosting models with Random Forest.
4. Lock one model before evaluating June.

## 2. Data and Features

- **Training:** February 2025-April 2026, 151,691 transactions.
- **Validation:** May 2026, 11,753 transactions.
- **Final test:** June 2026.
- **Features:** 38 interpretable predictors.

Numeric features use train-fitted imputation and scaling. Categories use train-only frequency encoding. Missing-value flags and target-derived predictors are not included.

## 3. Boosting Model Tuning

The light search tested tree depth, learning rate, early stopping, row and feature sampling, child size, and L1/L2 regularization.

| Model | Depth | Learning rate | Trees | May R2 | May MAPE | May MdAPE |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 6 | 0.05 | 1,495 | **0.908** | **12.18%** | **8.38%** |
| LightGBM | 6 | 0.05 | 1,500 | 0.904 | 12.38% | 8.60% |
| CatBoost | 6 | 0.05 | 1,500 | 0.903 | 12.54% | 8.74% |

**Result:** XGBoost produced the strongest overall validation performance.

## 4. Final Model Selection

| Model | May R2 | May MAE | May MAPE | May MdAPE | MdAPE gap |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.908** | **$155,112** | 12.18% | 8.38% | 0.79 pp |
| LightGBM | 0.904 | $158,205 | 12.38% | 8.60% | 0.73 pp |
| CatBoost | 0.903 | $159,446 | 12.54% | 8.74% | **0.65 pp** |
| Random Forest | 0.889 | $160,504 | **12.01%** | **7.73%** | 2.90 pp |

- **Accuracy:** XGBoost has the highest R2 and lowest MAE.
- **Tradeoff:** Random Forest's MdAPE is 0.65 percentage points lower.
- **Stability:** XGBoost's train-to-May MdAPE gap is much smaller than Random Forest's.

The MdAPE gap is validation MdAPE minus training MdAPE. A smaller gap indicates less deterioration on later data.

## 5. Locked XGBoost

| Hyperparameter | Final value |
|---|---:|
| Maximum depth | 6 |
| Learning rate | 0.05 |
| Trees selected by early stopping | 1,495 |
| Row sampling | 0.85 |
| Feature sampling | 0.85 |
| Minimum child weight | 1 |
| L1 / L2 | 0 / 1 |

These settings were locked before evaluating June.

## 6. Final June Test

| Model | R2 | MAE | RMSE | MAPE | MdAPE |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.911** | **$154,492** | **$284,502** | 12.17% | 8.46% |
| Random Forest | 0.891 | $160,592 | $314,448 | **12.01%** | **7.80%** |

- XGBoost retains its R2 and dollar-error advantage on June.
- Random Forest remains slightly better on typical percentage error.
- XGBoost provides the better overall balance for valuation and pricing review.

## 7. Feature Importance

The strongest XGBoost predictors are:

1. Bathrooms total.
2. Living area.
3. County frequency.
4. School-district county frequency.
5. School-district locale frequency.
6. Longitude and latitude.
7. ZIP-code frequency.

Property size and geography provide most of the predictive signal. Frequency-encoded variables are retained, while `_missing` and `_flag` variables are excluded. Importance represents predictive contribution, not causality.

## 8. Robustness

The main results focus on typical market transactions using limits learned from training data only. This keeps model development focused on the market segment where the model is intended to support pricing decisions.

When all June transactions are included:

| Population | R2 | MAPE | MdAPE |
|---|---:|---:|---:|
| All June transactions | 0.678 | 13.81% | 8.67% |

Performance falls because luxury and unusual properties are harder to predict. These cases should receive manual review rather than automatic pricing recommendations.

## 9. Conclusion

- **Final model:** XGBoost.
- **Why:** best balance of R2, dollar error, percentage error, and stability.
- **Business use:** identify potentially mispriced listings and support agent pricing review.
- **Limitation:** one validation month and one test month; unusual and high-value properties remain higher risk.

The notebook depends on `week7_modeling.py` for data preparation, model tuning, evaluation, and reproducibility.
