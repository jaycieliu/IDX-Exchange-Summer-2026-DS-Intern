from pathlib import Path
import sys

import numpy as np
import pandas as pd


SOURCE_DATA_PATH = "outputs/week3_preprocessing/crmls_sfr_quality_cleaned_202501_202606.csv"
OUTPUT_DIR = "outputs/week8_evaluation"
METRICS_PATH = "outputs/week8_evaluation/metrics_summary.csv"
PREDICTIONS_PATH = "outputs/week8_evaluation/week8_xgboost_june_predictions.csv"
PRICE_SEGMENT_ORDER = ["Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest"]
TARGET = "ClosePrice"


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ape = np.abs(predicted - actual) / actual
    residual = predicted - actual
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return {
        "rows": len(actual),
        "r2": 1 - ss_res / ss_tot,
        "mae": np.mean(np.abs(residual)),
        "rmse": np.sqrt(np.mean(residual**2)),
        "mape": np.mean(ape),
        "mdape": np.median(ape),
        "p90_ape": np.quantile(ape, 0.90),
        "median_error": np.median(residual),
    }


def _eligible_rows(frame):
    return (
        frame["PropertyType"].astype(str).str.strip().eq("Residential")
        & frame["PropertySubType"].astype(str).str.strip().eq("SingleFamilyResidence")
        & frame["CloseDate"].notna()
        & frame[TARGET].gt(0)
    )


def _load_cleaned_source(root):
    frame = pd.read_csv(Path(root) / SOURCE_DATA_PATH, low_memory=False)
    frame["CloseDate"] = pd.to_datetime(frame["CloseDate"], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame["close_month"] = pd.PeriodIndex(frame["close_month"].astype(str), freq="M").astype(str)
    return frame.loc[_eligible_rows(frame)].copy()


def _training_bounds(root):
    frame = _load_cleaned_source(root)
    month = pd.PeriodIndex(frame["close_month"], freq="M")
    train = frame.loc[
        (month >= pd.Period("2025-02")) & (month <= pd.Period("2026-04"))
    ].sort_values("CloseDate").reset_index(drop=True)

    price_low, price_high = train[TARGET].quantile([0.005, 0.995])
    price_in_range = train[TARGET].between(price_low, price_high)
    ppsf = pd.to_numeric(train["price_per_sqft_audit"], errors="coerce")
    ppsf_low, ppsf_high = ppsf[price_in_range].dropna().quantile([0.005, 0.995])
    return price_low, price_high, ppsf_low, ppsf_high


def _assign_train_defined_price_segment(values, root):
    bands_path = Path(root) / "outputs/week6_feature_engineering/week6_train_defined_price_bands.csv"
    bands = pd.read_csv(bands_path)
    edges = bands["lower_bound"].replace({"-inf": -np.inf, "inf": np.inf}).astype(float)
    last_upper = pd.to_numeric(bands["upper_bound"].replace({"inf": np.inf}), errors="coerce").iloc[-1]
    edges = list(edges) + [last_upper]
    labels = bands["price_segment"].tolist()
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True)


def build_xgboost_predictions(root="."):
    root = Path(root)
    sys.path.insert(0, str(root / "week7"))
    from week7_modeling import load_june, load_train_validation, _make_model

    data = load_train_validation(root)
    params = {
        "model": "XGBoost",
        "depth": 6,
        "learning_rate": 0.05,
        "row_sample": 0.85,
        "feature_sample": 0.85,
        "min_child": 1,
        "l1": 0.0,
        "l2": 1.0,
    }
    model = _make_model(params)
    model.fit(
        data["x_train"],
        data["y_train"],
        eval_set=[(data["x_validation"], data["y_validation"])],
        verbose=False,
    )
    june = load_june(root, data["state"], data["bounds"])
    prediction = model.predict(june["x"])

    source = _load_cleaned_source(root)
    june_source = (
        source.loc[source["close_month"].eq("2026-06")]
        .reset_index(drop=True)
    )

    if len(june_source) != len(prediction):
        raise ValueError(f"Prediction/source row mismatch: {len(prediction)} vs {len(june_source)}")

    predictions = june_source[
        [
            "CloseDate",
            "close_month",
            "ClosePrice",
            "City",
            "PostalCode",
            "CountyOrParish",
            "UnifiedSchoolDistrictName",
            "price_per_sqft_audit",
        ]
    ].copy()
    predictions["model"] = "Week 7 locked XGBoost"
    predictions["prediction"] = prediction
    predictions["error"] = predictions["prediction"] - predictions["ClosePrice"]
    predictions["absolute_error"] = predictions["error"].abs()
    predictions["absolute_percentage_error"] = (
        predictions["absolute_error"] / predictions["ClosePrice"]
    )
    predictions["split"] = "test"
    predictions["price_segment"] = _assign_train_defined_price_segment(
        predictions["ClosePrice"], root
    )

    price_low, price_high, ppsf_low, ppsf_high = _training_bounds(root)
    ppsf = predictions["price_per_sqft_audit"]
    predictions["is_june_in_range"] = (
        predictions[TARGET].between(price_low, price_high)
        & (ppsf.isna() | ppsf.between(ppsf_low, ppsf_high))
    )
    predictions.to_csv(root / PREDICTIONS_PATH, index=False)
    return predictions


def load_june_populations(root):
    predictions = build_xgboost_predictions(root)
    predictions["is_june_in_range"] = predictions["is_june_in_range"].astype(bool)
    return {
        "june_in_range": predictions.loc[predictions["is_june_in_range"]].copy(),
        "full_june_robustness": predictions.copy(),
    }


def build_metrics_summary(root="."):
    root = Path(root)
    populations = load_june_populations(root)
    predictions = populations["june_in_range"]
    full_june = populations["full_june_robustness"]

    rows = []
    overall = _metrics(predictions["ClosePrice"], predictions["prediction"])
    rows.append({"section": "main", "price_segment": "June in-range", **overall})

    available_segments = [
        segment for segment in PRICE_SEGMENT_ORDER
        if segment in set(predictions["price_segment"])
    ]
    for segment in available_segments:
        frame = predictions.loc[predictions["price_segment"].eq(segment)]
        segment_metrics = _metrics(frame["ClosePrice"], frame["prediction"])
        segment_metrics["median_close_price"] = frame["ClosePrice"].median()
        rows.append(
            {
                "section": "main_price_band",
                "price_segment": segment,
                **segment_metrics,
            }
        )

    robustness = _metrics(full_june["ClosePrice"], full_june["prediction"])
    rows.append({
        "section": "robustness",
        "price_segment": "Full June",
        **robustness,
    })

    summary = pd.DataFrame(rows)
    ordered_columns = [
        "section",
        "price_segment",
        "rows",
        "median_close_price",
        "r2",
        "mae",
        "rmse",
        "mape",
        "mdape",
        "p90_ape",
        "median_error",
    ]
    summary = summary.reindex(columns=ordered_columns)

    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / METRICS_PATH, index=False)
    return summary


def readable_summary(metrics_summary):
    overall = metrics_summary.loc[metrics_summary["section"].eq("main")].iloc[0]
    robustness = metrics_summary.loc[metrics_summary["section"].eq("robustness")].iloc[0]
    segments = metrics_summary.loc[metrics_summary["section"].eq("main_price_band")].copy()
    best = segments.sort_values(["mdape", "mape"]).iloc[0]
    worst = segments.sort_values(["mdape", "mape"], ascending=False).iloc[0]
    return {
        "overall_r2": overall["r2"],
        "overall_mape": overall["mape"],
        "overall_mdape": overall["mdape"],
        "robustness_mape": robustness["mape"],
        "robustness_mdape": robustness["mdape"],
        "best_segment": best["price_segment"],
        "best_mdape": best["mdape"],
        "worst_segment": worst["price_segment"],
        "worst_mdape": worst["mdape"],
        "worst_mape": worst["mape"],
    }


if __name__ == "__main__":
    table = build_metrics_summary(Path.cwd())
    print(table.round({"r2": 3, "mape": 3, "mdape": 3, "p90_ape": 3}).to_string(index=False))
