from itertools import product
from pathlib import Path
import gc

import lightgbm
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


TARGET = "ClosePrice"
RANDOM_STATE = 42
TRAIN_START = "2025-02"
TRAIN_END = "2026-04"
VALIDATION_MONTH = "2026-05"
TEST_MONTH = "2026-06"
EARLY_STOPPING_ROUNDS = 75
MAX_TREES = 1500

NUMERIC_FEATURES = [
    "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet",
    "LotSizeAcres", "LotSizeArea", "YearBuilt", "property_age_at_close",
    "bed_bath_ratio", "living_area_to_lot_ratio", "Latitude", "Longitude",
    "GarageSpaces", "ParkingTotal", "AssociationFee", "Stories", "MainLevelBedrooms",
    "UnifiedSchoolDistrictEnrollTotal", "UnifiedSchoolDistrictAreaSqMi",
    "UnifiedSchoolDistrictEnrollmentDensity", "close_month_sin", "close_month_cos",
]
CATEGORICAL_FEATURES = [
    "City", "PostalCode", "CountyOrParish", "MLSAreaMajor", "HighSchoolDistrict",
    "UnifiedSchoolDistrictName", "UnifiedSchoolDistrictCounty",
    "UnifiedSchoolDistrictLocaleDesc", "Levels", "AssociationFeeFrequency",
    "StateOrProvince",
]
BINARY_FEATURES = [
    "ViewYN", "PoolPrivateYN", "AttachedGarageYN", "FireplaceYN", "NewConstructionYN",
]
BINARY_MAP = {
    "Y": 1.0, "YES": 1.0, "TRUE": 1.0, "T": 1.0, "1": 1.0, "1.0": 1.0,
    "N": 0.0, "NO": 0.0, "FALSE": 0.0, "F": 0.0, "0": 0.0, "0.0": 0.0,
}


def evaluate(actual, predicted):
    ape = np.abs(predicted - actual) / actual
    return {
        "R2": r2_score(actual, predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE": ape.mean(),
        "MdAPE": np.median(ape),
    }


def _prepare_features(frame):
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    prepared = frame.loc[:, columns].copy()
    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        values = prepared[column].astype("string").str.strip().replace({"": pd.NA})
        if column == "PostalCode":
            values = values.str.replace(r"\.0$", "", regex=True)
        prepared[column] = values.astype(object).where(values.notna(), np.nan)
    for column in BINARY_FEATURES:
        values = prepared[column].astype("string").str.strip().str.upper()
        prepared[column] = values.map(BINARY_MAP).astype(float)
    return prepared


def _fit_preprocessor(frame):
    state = {"numeric": {}, "category": {}, "binary": {}}
    for column in NUMERIC_FEATURES:
        values = frame[column]
        filled = values.fillna(values.median())
        state["numeric"][column] = {
            "median": values.median(),
            "mean": filled.mean(),
            "std": filled.std(ddof=0) or 1.0,
        }
    for column in CATEGORICAL_FEATURES:
        values = frame[column]
        mode = values.dropna().mode().iloc[0] if values.notna().any() else "Unknown"
        filled = values.where(values.notna(), mode)
        state["category"][column] = {
            "mode": mode,
            "frequency": filled.value_counts(normalize=True).to_dict(),
        }
    for column in BINARY_FEATURES:
        mode = frame[column].mode()
        state["binary"][column] = float(mode.iloc[0]) if len(mode) else 0.0
    return state


def _transform_features(frame, state):
    output = {}
    for column in NUMERIC_FEATURES:
        params = state["numeric"][column]
        values = frame[column].fillna(params["median"])
        output[column] = (values - params["mean"]) / params["std"]
    for column in CATEGORICAL_FEATURES:
        params = state["category"][column]
        values = frame[column].where(frame[column].notna(), params["mode"])
        output[f"{column}_frequency"] = values.map(params["frequency"]).fillna(0.0)
    for column in BINARY_FEATURES:
        output[column] = frame[column].fillna(state["binary"][column])
    return pd.DataFrame(output, index=frame.index)


def _eligible_rows(frame):
    return (
        frame["PropertyType"].astype(str).str.strip().eq("Residential")
        & frame["PropertySubType"].astype(str).str.strip().eq("SingleFamilyResidence")
        & frame["CloseDate"].notna()
        & frame[TARGET].gt(0)
    )


def load_train_validation(root):
    path = Path(root) / "outputs/week3_preprocessing/crmls_sfr_quality_cleaned_202501_202606.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame["CloseDate"] = pd.to_datetime(frame["CloseDate"], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame["close_month"] = pd.PeriodIndex(frame["close_month"].astype(str), freq="M").astype(str)
    frame = frame.loc[_eligible_rows(frame)].copy()

    month = pd.PeriodIndex(frame["close_month"], freq="M")
    train_raw = frame.loc[
        (month >= pd.Period(TRAIN_START)) & (month <= pd.Period(TRAIN_END))
    ].sort_values("CloseDate").reset_index(drop=True)
    validation_full = frame.loc[month == pd.Period(VALIDATION_MONTH)].reset_index(drop=True)

    price_low, price_high = train_raw[TARGET].quantile([0.005, 0.995])
    price_in_range = train_raw[TARGET].between(price_low, price_high)
    ppsf = pd.to_numeric(train_raw["price_per_sqft_audit"], errors="coerce")
    ppsf_low, ppsf_high = ppsf[price_in_range].dropna().quantile([0.005, 0.995])
    bounds = (price_low, price_high, ppsf_low, ppsf_high)

    train = train_raw.loc[_in_training_range(train_raw, bounds)].reset_index(drop=True)
    validation = validation_full.loc[_in_training_range(validation_full, bounds)].reset_index(drop=True)
    state = _fit_preprocessor(_prepare_features(train))

    return {
        "x_train": _transform_features(_prepare_features(train), state),
        "y_train": train[TARGET].to_numpy(float),
        "x_validation": _transform_features(_prepare_features(validation), state),
        "y_validation": validation[TARGET].to_numpy(float),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "state": state,
        "bounds": bounds,
    }


def _in_training_range(frame, bounds):
    price_low, price_high, ppsf_low, ppsf_high = bounds
    ppsf = pd.to_numeric(frame["price_per_sqft_audit"], errors="coerce")
    return (
        frame[TARGET].between(price_low, price_high)
        & (ppsf.isna() | ppsf.between(ppsf_low, ppsf_high))
    )


def search_space():
    base = {
        "XGBoost": {"min_child": 1, "l1": 0.0, "l2": 1.0},
        "LightGBM": {"min_child": 20, "l1": 0.0, "l2": 1.0},
        "CatBoost": {"min_child": 20, "l1": np.nan, "l2": 3.0},
    }
    candidates = []
    for family, settings in base.items():
        for depth, rate in product([3, 4, 6], [0.03, 0.05]):
            candidates.append({
                "model": family,
                "depth": depth,
                "learning_rate": rate,
                "row_sample": 0.85,
                "feature_sample": 0.85,
                **settings,
            })
        for row_sample, feature_sample, l1 in [(0.70, 0.90, 1.0), (0.90, 0.90, 0.0)]:
            candidates.append({
                "model": family,
                "depth": 6,
                "learning_rate": 0.05,
                "row_sample": row_sample,
                "feature_sample": feature_sample,
                "min_child": settings["min_child"],
                "l1": np.nan if family == "CatBoost" else l1,
                "l2": 10.0,
            })
    return candidates


def _make_model(params):
    if params["model"] == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror", tree_method="hist", eval_metric="mape",
            n_estimators=MAX_TREES, max_depth=params["depth"],
            learning_rate=params["learning_rate"], subsample=params["row_sample"],
            colsample_bytree=params["feature_sample"], min_child_weight=params["min_child"],
            reg_alpha=params["l1"], reg_lambda=params["l2"],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
    if params["model"] == "LightGBM":
        return LGBMRegressor(
            n_estimators=MAX_TREES, max_depth=params["depth"],
            num_leaves=min(2 ** params["depth"] - 1, 63),
            learning_rate=params["learning_rate"], subsample=params["row_sample"],
            subsample_freq=1, colsample_bytree=params["feature_sample"],
            min_child_samples=params["min_child"], reg_alpha=params["l1"],
            reg_lambda=params["l2"], random_state=RANDOM_STATE, n_jobs=-1,
            verbosity=-1, deterministic=True, force_col_wise=True,
        )
    return CatBoostRegressor(
        iterations=MAX_TREES, depth=params["depth"], learning_rate=params["learning_rate"],
        loss_function="RMSE", eval_metric="MAPE", grow_policy="Depthwise",
        bootstrap_type="Bernoulli", subsample=params["row_sample"],
        rsm=params["feature_sample"], min_data_in_leaf=params["min_child"],
        l2_leaf_reg=params["l2"], random_strength=1.0, random_seed=RANDOM_STATE,
        thread_count=-1, verbose=False, allow_writing_files=False,
    )


def _fit_boosting(params, data):
    model = _make_model(params)
    if params["model"] == "XGBoost":
        model.fit(
            data["x_train"], data["y_train"],
            eval_set=[(data["x_validation"], data["y_validation"])], verbose=False,
        )
        trees = model.best_iteration + 1
    elif params["model"] == "LightGBM":
        model.fit(
            data["x_train"], data["y_train"],
            eval_set=[(data["x_validation"], data["y_validation"])], eval_metric="mape",
            callbacks=[lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        trees = model.best_iteration_
    else:
        model.fit(
            data["x_train"], data["y_train"],
            eval_set=(data["x_validation"], data["y_validation"]),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True, verbose=False,
        )
        trees = model.tree_count_

    train_metrics = evaluate(data["y_train"], model.predict(data["x_train"]))
    validation_metrics = evaluate(
        data["y_validation"], model.predict(data["x_validation"])
    )
    result = {
        **params,
        "trees": int(trees),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        "gap": validation_metrics["MdAPE"] - train_metrics["MdAPE"],
    }
    return model, result


def tune_boosting_models(data):
    rows = []
    best_models = {}
    for family in ["XGBoost", "LightGBM", "CatBoost"]:
        best_score = None
        for params in [p for p in search_space() if p["model"] == family]:
            model, result = _fit_boosting(params, data)
            rows.append(result)
            score = (
                result["validation_MdAPE"],
                result["validation_MAPE"],
                -result["validation_R2"],
            )
            if best_score is None or score < best_score:
                best_score = score
                best_models[family] = model
            else:
                del model
            gc.collect()
    results = pd.DataFrame(rows)
    best = (
        results.sort_values(
            ["validation_MdAPE", "validation_MAPE", "validation_R2"],
            ascending=[True, True, False],
        )
        .groupby("model", as_index=False)
        .first()
    )
    return results, best, best_models


def validation_comparison(data, best_results, best_models):
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE,
    ).fit(data["x_train"], data["y_train"])
    rows = []
    for name, model in {**best_models, "Random Forest": rf_model}.items():
        train_metrics = evaluate(data["y_train"], model.predict(data["x_train"]))
        validation_metrics = evaluate(
            data["y_validation"], model.predict(data["x_validation"])
        )
        rows.append({
            "Model": name,
            "R2": validation_metrics["R2"],
            "MAE": validation_metrics["MAE"],
            "MAPE": validation_metrics["MAPE"] * 100,
            "MdAPE": validation_metrics["MdAPE"] * 100,
            "Gap": (validation_metrics["MdAPE"] - train_metrics["MdAPE"]) * 100,
        })
    return pd.DataFrame(rows), rf_model


def load_june(root, state, bounds):
    path = Path(root) / "outputs/week3_preprocessing/crmls_sfr_quality_cleaned_202501_202606.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame["CloseDate"] = pd.to_datetime(frame["CloseDate"], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame["close_month"] = pd.PeriodIndex(frame["close_month"].astype(str), freq="M").astype(str)
    frame = frame.loc[_eligible_rows(frame)].copy()
    month = pd.PeriodIndex(frame["close_month"], freq="M")
    june = frame.loc[month == pd.Period(TEST_MONTH)].reset_index(drop=True)
    mask = _in_training_range(june, bounds).to_numpy()
    return {
        "x": _transform_features(_prepare_features(june), state),
        "y": june[TARGET].to_numpy(float),
        "primary_mask": mask,
    }


def june_comparison(june, models):
    rows = []
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(june["x"])
        predictions[name] = prediction
        metrics = evaluate(
            june["y"][june["primary_mask"]], prediction[june["primary_mask"]]
        )
        rows.append({
            "Model": name,
            "R2": metrics["R2"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"] * 100,
            "MdAPE": metrics["MdAPE"] * 100,
        })
    return pd.DataFrame(rows), predictions
