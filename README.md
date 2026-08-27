# California Residential Price Prediction

## Project Overview

This project builds an end-to-end machine learning workflow to estimate the final sale price of California single-family residential homes using historical CRMLS sold-property data.

The goal is not to replace real estate judgment. The goal is to support better pricing review, valuation quality checks, and manual-review prioritization.

The final model can help answer:

- Does a sale or listing price look reasonable?
- Which properties should be reviewed more carefully?
- Which price ranges or locations have higher prediction risk?
- How can historical market data support pricing decisions?

## Final Result

The final model selected for the main evaluation is **XGBoost**.

On the primary June 2026 test set, the model achieved a typical prediction error of about **8.5%**.

| Evaluation view | Homes reviewed | Overall fit | Average error | Typical error |
|---|---:|---:|---:|---:|
| Main June test set | 12,566 | 0.911 | 12.17% | 8.46% |
| Full June robustness check | 12,851 | 0.678 | 13.81% | 8.67% |

Plain-English read:

- **Overall fit 0.911** means the model captures most of the price pattern in the main test group.
- **Typical error 8.46%** means a normal prediction miss is around 8-9%.
- **Average error 12.17%** is higher because difficult properties create larger misses.
- Full June performance is weaker because it includes more unusual records, so those cases need manual review.

## What I Built

This project covers the full data science workflow:

1. Collected and organized CRMLS sold-property data.
2. Explored the target variable and key property fields.
3. Cleaned the raw monthly files into a modeling dataset.
4. Added geospatial and school-district context.
5. Built baseline and advanced machine learning models.
6. Compared models with business-relevant error metrics.
7. Evaluated model reliability by price range.
8. Created a Streamlit web app to present the results and demonstrate prediction.
9. Documented assumptions, limitations, and reproducibility steps.

## Data Used

The project uses historical California CRMLS sold-property records.

Each record includes information such as:

- final sale price,
- living area,
- bedrooms and bathrooms,
- lot size,
- property age,
- garage and parking details,
- city, ZIP code, county, and coordinates,
- listing and close-date information.

Only these homes are included:

- Residential properties
- Single-family residences
- California sold transactions

School-district boundary data was also added to provide more location context.

## Data Preparation

The raw monthly files were cleaned and combined into one model-ready dataset.

Main preparation work:

- filtered the data to the correct property type,
- checked missing values and invalid property records,
- created quality flags for unusual values,
- handled missing values using training-data rules,
- converted categorical location fields into model-ready features,
- added school-district information from property location,
- created time-based train, validation, and test splits.

The time-based split is important because real estate markets change over time. The model was trained on earlier sales and tested on later sales, which is closer to how it would be used in practice.

## Feature Engineering

The model uses practical pricing signals that are understandable for real estate review:

- home size,
- lot size,
- bedrooms and bathrooms,
- property age,
- bed/bath ratio,
- living-area-to-lot ratio,
- city, ZIP code, county, and coordinates,
- school-district context,
- amenities such as pool, view, fireplace, attached garage, and new construction,
- close-month seasonality.

Fields that could leak the answer were excluded. For example, the model should not rely on fields that directly reveal or are derived from final close price.

## Modeling Work

The modeling process started with a simple baseline and then moved to stronger tree-based methods.

| Model stage | Purpose | Result |
|---|---|---|
| Linear Regression | Establish a simple baseline | Useful benchmark, but error was too high for pricing review |
| Decision Tree / Random Forest | Capture non-linear property and location patterns | Strong improvement over the linear baseline |
| XGBoost / LightGBM / CatBoost | Test advanced gradient boosting models | XGBoost had the best overall balance |

The final model was selected because it provided the strongest balance of:

- overall fit,
- dollar error,
- percentage error,
- stability from validation to test,
- usefulness for real estate pricing review.

## Where The Model Works Best

The model performs best on middle-price homes. These are the safest use case for pricing review.

| Price range group | Homes reviewed | Median sale price | Average error | Typical error |
|---|---:|---:|---:|---:|
| Lowest-price group | 2,483 | $451,500 | 14.33% | 8.95% |
| Lower-middle group | 2,354 | $688,944 | 10.30% | 6.79% |
| Middle group | 2,434 | $900,000 | 10.64% | 7.49% |
| Upper-middle group | 2,581 | $1,285,000 | 11.94% | 8.80% |
| Highest-price group | 2,714 | $2,220,000 | 13.39% | 10.88% |

Key findings:

1. **Middle-price homes are the strongest segment.** Lower-middle and middle-price homes have the lowest typical errors.
2. **Lowest-price homes need caution.** Small dollar misses can become large percentage errors.
3. **Highest-price homes need manual review.** Luxury and unusual homes are harder to estimate reliably.

## Business Recommendation

Use the model as a first-pass pricing review tool.

Recommended workflow:

1. Score a property or group of properties.
2. Compare the predicted price with the listed or closed price.
3. Flag large gaps for manual review.
4. Use comparable sales and local market judgment before making a final pricing decision.

The model is useful for decision support, but it should not be used as a standalone pricing engine.

## Interactive Web Demo

A Streamlit dashboard was created to make the project easier to review and present.

The app is designed for a mentor, manager, or stakeholder who wants to understand the project without opening the notebooks.

| View | What it shows | Why it matters |
|---|---|---|
| Overview | Main model result and business interpretation | Gives a quick executive readout |
| Predict | A simple property price-estimation form | Demonstrates how the model could support one-property review |
| Geographic Analysis | County and city-level error summaries | Shows where predictions may need more review |
| Market Trends | Monthly volume, median sale price, and price per square foot | Adds market context |
| Model Performance | Error by price range | Explains where the model is strongest and weakest |
| Handoff | Run instructions and production cautions | Supports future project continuation |

To open the app locally:

```bash
source .venv/bin/activate
streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

Important app note:

- The dashboard report pages use the final evaluation outputs.
- The prediction page uses the saved deployable Random Forest pipeline because that is the available joblib model artifact.
- The final project conclusion still uses XGBoost as the best evaluated model.
- A production version should save and deploy the final XGBoost artifact so the app and final report use the same model.

## Main Limitations

- The model was evaluated on one final test month, so more future-month testing is needed.
- Luxury, unusual, and very low-price homes still have higher prediction risk.
- Location features help, but they do not fully replace neighborhood expertise.
- School-district enrichment is useful, but needs more geospatial quality checks before production use.
- The model predicts historical final sale price, not a guaranteed listing price.
- Feature importance should be read as model behavior, not causal proof.

## Deliverables

Main project materials include:

- exploratory data analysis notebook,
- preprocessing notebook and cleaned dataset,
- baseline model notebook,
- model comparison notebook,
- feature engineering and pipeline selection notebook,
- advanced model comparison notebook,
- final evaluation notebook and metrics summary,
- Streamlit web dashboard,
- README documentation and reproducibility notes.

## Reproducibility Notes

To rerun the project, create a Python environment and install the modeling packages:

```bash
cd "/Users/amyliu/Desktop/summer intern"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn joblib xgboost lightgbm catboost matplotlib seaborn notebook streamlit
```

Then rebuild the main artifacts:

```bash
python scripts/build_week3_preprocessing_deliverable.py
python scripts/build_week6_feature_engineering_notebook.py
python scripts/build_week7_advanced_models_notebook.py
python scripts/build_week8_evaluation_notebook.py
python scripts/week8_evaluation.py
```

## Final Takeaway

This project produced a complete machine learning workflow for California residential price prediction: raw CRMLS data preparation, feature engineering, model comparison, final evaluation, segment-risk analysis, and an interactive web dashboard.

The final XGBoost model is strong enough to support pricing review and valuation QA, especially for typical middle-market homes. It should be used to prioritize human review, not to replace professional pricing judgment.
