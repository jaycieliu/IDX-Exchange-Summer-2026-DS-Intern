import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


REQUIRED_RAW_COLUMNS = [
    "LivingArea",
    "LotSizeSquareFeet",
    "BedroomsTotal",
    "BathroomsTotalInteger",
]

BINARY_MAP = {
    "Y": 1.0,
    "YES": 1.0,
    "TRUE": 1.0,
    "T": 1.0,
    "1": 1.0,
    "1.0": 1.0,
    "N": 0.0,
    "NO": 0.0,
    "FALSE": 0.0,
    "F": 0.0,
    "0": 0.0,
    "0.0": 0.0,
}


def numeric_column(frame, column):
    source = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(source, errors="coerce")


def normalize_categorical_column(frame, column):
    source = frame[column] if column in frame.columns else pd.Series(pd.NA, index=frame.index)
    values = source.astype("string").str.strip().replace({"": pd.NA})
    if column == "PostalCode":
        values = values.str.replace(r"\.0$", "", regex=True)
    return values


def normalize_binary_column(frame, column):
    source = frame[column] if column in frame.columns else pd.Series(pd.NA, index=frame.index)
    normalized = source.astype("string").str.strip().str.upper().replace({"": pd.NA})
    values = normalized.map(BINARY_MAP)
    invalid = sorted(normalized[normalized.notna() & values.isna()].dropna().unique().tolist())
    if invalid:
        sample = invalid[:5]
        raise ValueError(f"Unrecognized binary values in {column}: {sample}")
    return values.astype(float)


def require_raw_columns(frame):
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")


def add_engineered_features(frame):
    out = frame.copy()
    require_raw_columns(out)

    living_area = numeric_column(out, "LivingArea")
    lot_size = numeric_column(out, "LotSizeSquareFeet")
    bedrooms = numeric_column(out, "BedroomsTotal")
    bathrooms = numeric_column(out, "BathroomsTotalInteger")
    garage_spaces = numeric_column(out, "GarageSpaces")
    fireplaces = numeric_column(out, "FireplacesTotal")
    association_fee = numeric_column(out, "AssociationFee")
    tax_annual = numeric_column(out, "TaxAnnualAmount")

    out["log_living_area"] = np.log1p(living_area.where(living_area > 0))
    out["log_lot_size_sqft"] = np.log1p(lot_size.where(lot_size > 0))
    out["bed_bath_ratio"] = bedrooms / bathrooms.replace(0, np.nan)
    out["bath_per_bedroom"] = bathrooms / bedrooms.replace(0, np.nan)
    out["total_bed_bath"] = bedrooms + bathrooms
    out["garage_present"] = garage_spaces.fillna(0).gt(0).astype(float)
    out["garage_per_bedroom"] = garage_spaces / bedrooms.replace(0, np.nan)
    out["fireplaces_per_bedroom"] = fireplaces / bedrooms.replace(0, np.nan)
    out["association_fee_present"] = association_fee.fillna(0).gt(0).astype(float)
    out["tax_per_living_sqft"] = tax_annual / living_area.replace(0, np.nan)
    return out


class CRMLSFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_spec, missingness_limit=0.90):
        self.feature_spec = feature_spec
        self.missingness_limit = missingness_limit

    def fit(self, X, y=None):
        frame = add_engineered_features(X)
        self.numeric_features_ = []
        self.categorical_features_ = []
        self.binary_features_ = []
        self.numeric_medians_ = {}
        self.numeric_means_ = {}
        self.numeric_stds_ = {}
        self.categorical_frequency_maps_ = {}
        self.binary_fill_values_ = {}
        self.excluded_numeric_ = []
        self.excluded_categorical_ = []
        self.excluded_binary_ = []
        self.constant_feature_names_ = []

        if "UnifiedSchoolDistrictName" in frame.columns:
            district_values = normalize_categorical_column(
                frame, "UnifiedSchoolDistrictName"
            ).fillna("Unknown")
            self.school_district_frequency_map_ = (
                district_values.value_counts(normalize=True).astype(float).to_dict()
            )
        else:
            self.school_district_frequency_map_ = {}

        frame = self._add_train_frequency(frame)

        for col in self.feature_spec["numeric"]:
            if col not in frame.columns:
                self.excluded_numeric_.append(col)
                continue
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.notna().sum() == 0 or values.isna().mean() > self.missingness_limit:
                self.excluded_numeric_.append(col)
                continue
            median = values.median()
            filled = values.fillna(median)
            std = filled.std(ddof=0)
            self.numeric_features_.append(col)
            self.numeric_medians_[col] = float(median)
            self.numeric_means_[col] = float(filled.mean())
            self.numeric_stds_[col] = float(std) if std and np.isfinite(std) else 1.0

        for col in self.feature_spec["categorical"]:
            if col not in frame.columns:
                self.excluded_categorical_.append(col)
                continue
            values = normalize_categorical_column(frame, col)
            if values.isna().mean() > self.missingness_limit:
                self.excluded_categorical_.append(col)
                continue
            filled = values.fillna("Unknown")
            self.categorical_features_.append(col)
            self.categorical_frequency_maps_[col] = (
                (filled.value_counts() / len(filled)).astype(float).to_dict()
            )

        for col in self.feature_spec["binary"]:
            if col not in frame.columns:
                self.excluded_binary_.append(col)
                continue
            values = normalize_binary_column(frame, col)
            mode = values.mode()
            self.binary_features_.append(col)
            self.binary_fill_values_[col] = float(mode.iloc[0]) if len(mode) else 0.0

        candidate_feature_names = self._feature_names()
        transformed_train = self._transform_prepared_frame(frame, candidate_feature_names)
        self.constant_feature_names_ = [
            col for col in transformed_train.columns
            if transformed_train[col].nunique(dropna=False) <= 1
        ]
        self.feature_names_out_ = [
            col for col in candidate_feature_names
            if col not in self.constant_feature_names_
        ]
        return self

    def transform(self, X):
        frame = self._add_train_frequency(add_engineered_features(X))
        return self._transform_prepared_frame(frame, self.feature_names_out_)

    def _transform_prepared_frame(self, frame, feature_names):
        columns = {}

        for col in self.numeric_features_:
            source = frame[col] if col in frame.columns else pd.Series(pd.NA, index=frame.index)
            values = pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan)
            columns[f"{col}_was_missing"] = values.isna().astype(float).to_numpy()
            columns[f"{col}_scaled"] = (
                (values.fillna(self.numeric_medians_[col]) - self.numeric_means_[col])
                / self.numeric_stds_[col]
            ).to_numpy()

        for col in self.categorical_features_:
            values = normalize_categorical_column(frame, col)
            was_missing = values.isna()
            filled = values.fillna("Unknown")
            frequency_map = self.categorical_frequency_maps_[col]
            columns[f"{col}_frequency"] = filled.map(frequency_map).fillna(0).astype(float).to_numpy()
            columns[f"{col}_was_missing"] = was_missing.astype(float).to_numpy()

        for col in self.binary_features_:
            values = normalize_binary_column(frame, col)
            columns[col] = values.fillna(self.binary_fill_values_[col]).astype(float).to_numpy()

        return pd.DataFrame(columns, index=frame.index).reindex(columns=feature_names, fill_value=0)

    def get_feature_names_out(self):
        return np.asarray(self.feature_names_out_, dtype=object)

    def get_preprocess_params(self):
        return {
            "numeric_medians": self.numeric_medians_,
            "numeric_means": self.numeric_means_,
            "numeric_stds": self.numeric_stds_,
            "categorical_frequency_maps": self.categorical_frequency_maps_,
            "binary_fill_values": self.binary_fill_values_,
            "school_district_frequency_map": self.school_district_frequency_map_,
            "excluded_numeric": self.excluded_numeric_,
            "excluded_categorical": self.excluded_categorical_,
            "excluded_binary": self.excluded_binary_,
            "constant_feature_names": self.constant_feature_names_,
        }

    def _add_train_frequency(self, frame):
        out = frame.copy()
        if "UnifiedSchoolDistrictName" in out.columns:
            values = normalize_categorical_column(
                out, "UnifiedSchoolDistrictName"
            ).fillna("Unknown")
            out["school_district_train_frequency"] = (
                values.map(self.school_district_frequency_map_).fillna(0.0).astype(float)
            )
        elif "school_district_train_frequency" not in out.columns:
            out["school_district_train_frequency"] = 0.0
        return out

    def _feature_names(self):
        names = []
        for col in self.numeric_features_:
            names.extend([f"{col}_was_missing", f"{col}_scaled"])
        for col in self.categorical_features_:
            names.extend([f"{col}_frequency", f"{col}_was_missing"])
        names.extend(self.binary_features_)
        return names


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        self.selected_features_ = list(self.selected_features)
        return self

    def transform(self, X):
        return X.reindex(columns=self.selected_features_, fill_value=0)

    def get_feature_names_out(self):
        return np.asarray(self.selected_features_, dtype=object)
