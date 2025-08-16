from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, FunctionTransformer

# -----------------------------
# 1) Helpers: education binning & interaction features
# -----------------------------

EDU_BIN_ORDER = ["PreHighSchool", "IncompleteHS", "HighSchool", "Associate", "Bachelors", "Advanced"]

def map_education_to_bin(s: pd.Series) -> pd.Series:
    """Map raw 'education' to requested bins."""
    s = s.astype(str).str.strip()
    pre_hs = {"Preschool", "1st-4th", "5th-6th", "7th-8th"}
    incomplete_hs = {"9th", "10th", "11th"}
    hs = {"12th", "HS-grad"}
    assoc = {"Assoc-voc", "Assoc-acdm", "Some-college"}  # treat 'Some-college' as post‑HS coursework
    bach = {"Bachelors"}
    adv = {"Masters", "Prof-school", "Doctorate"}

    def _bin(x: str) -> str:
        if x in pre_hs:
            return "PreHighSchool"
        if x in incomplete_hs:
            return "IncompleteHS"
        if x in hs:
            return "HighSchool"
        if x in assoc:
            return "Associate"
        if x in bach:
            return "Bachelors"
        if x in adv:
            return "Advanced"
        # fallback
        return "HighSchool"

    return s.map(_bin).astype("category")


class InitialCleaner(BaseEstimator, TransformerMixin):
    """
    - Replace '.' with '_' in column names
    - Strip whitespace in string cells
    - Convert literal '?' to NaN and drop resulting NaN rows
    - Replace hyphens with spaces in native_country
    - Drop columns: fnlwgt, relationship, capital_loss - high correlation with better columns or low correlation with target
    - Add: married_together, sex_bin, capital_gain_bin, education_bin
    - Add interactions: pairwise products of age, hours_per_week, education_num
    NOTE: does not touch target y; use prepare_y() for that.
    """
    def __init__(self):
        self.columns_to_drop = ["fnlwgt", "relationship", "capital_loss"]
        self.interaction_cols = ["age", "hours_per_week", "education_num"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1) normalize column names
        X = X.copy()
        X.columns = [c.replace(".", "_").strip() for c in X.columns]

        # 2) strip string cells & standardize '?'
        for c in X.columns:
            if pd.api.types.is_string_dtype(X[c]) or pd.api.types.is_object_dtype(X[c]):
                X[c] = X[c].astype(str).str.strip()
                # Use NA for literal '?'
                X.loc[X[c] == "?", c] = np.nan

        # 3) drop rows with any NaN produced by '?' (and pre-existing NaN if any)
        X = X.dropna(axis=0, how="any").reset_index(drop=True)

        # 4) fix hyphens in native_country names (after rename)
        if "native_country" in X.columns:
            X["native_country"] = X["native_country"].str.replace("-", " ", regex=False)

        # 5) drop columns
        drop_these = [c for c in self.columns_to_drop if c in X.columns]
        X = X.drop(columns=drop_these, errors="ignore")

        # 6) marital feature
        if "marital_status" in X.columns:
            ms = X["marital_status"].astype(str)
        elif "marital.status" in X.columns:  # safety
            ms = X["marital.status"].astype(str)
        else:
            ms = pd.Series("", index=X.index)
        X["married_together"] = ms.isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(int)

        # 7) binary encodings requested (as features)
        if "sex" in X.columns:
            X["sex_bin"] = (X["sex"].astype(str).str.lower() == "male").astype(int)
        if "capital_gain" in X.columns:
            # keep raw numeric column for MinMax later
            X["capital_gain_bin"] = (pd.to_numeric(X["capital_gain"], errors="coerce") > 0).astype(int)

        # 8) education bin (categorical label to ordinal later)
        if "education" in X.columns:
            X["education_bin"] = map_education_to_bin(X["education"])
        else:
            # create neutral category if missing
            X["education_bin"] = pd.Categorical(["HighSchool"] * len(X), categories=EDU_BIN_ORDER)

        # 9) interaction features (pairwise)
        for col in self.interaction_cols:
            if col not in X.columns:
                raise ValueError(f"Expected column '{col}' not found for interactions.")
        # Coerce to numeric
        for col in self.interaction_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X["age_x_hours_per_week"] = X["age"] * X["hours_per_week"]
        X["age_x_education_num"] = X["age"] * X["education_num"]
        X["hours_per_week_x_education_num"] = X["hours_per_week"] * X["education_num"]

        return X


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer given cleaned/augmented columns.
    """
    # Columns for specific transforms
    onehot_cols = [c for c in ["workclass", "occupation", "race"] if c in feature_frame.columns]

    ordinal_col = ["education_bin"] if "education_bin" in feature_frame.columns else []

    minmax_cols = ["capital_gain"] if "capital_gain" in feature_frame.columns else []

    # numeric columns to standard-scale (exclude those with special handling)
    exclude_from_standard = set(onehot_cols + ordinal_col + minmax_cols)
    # remove raw categorical cols and text/object cols
    numeric_candidates = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    # Avoid double-scaling 'capital_gain' (it will get MinMax), keep binary features here as well
    numeric_std_cols = [c for c in numeric_candidates if c not in minmax_cols]

    # Ordinal order for education_bin
    ordinal_encoder = OrdinalEncoder(categories=[EDU_BIN_ORDER], dtype=np.int64)

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_cols),
            ("edu_ord", ordinal_encoder, ordinal_col),
            ("mm_capgain", MinMaxScaler(), minmax_cols),
            ("num_std", StandardScaler(), numeric_std_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )
    return preprocessor


# -----------------------------
# 2) Public API
# -----------------------------

def prepare_y(df: pd.DataFrame) -> pd.Series:
    """
    Binary encode target 'income' AFTER cleaning '?' rows, etc.
    1 if >50K else 0
    """
    cols = [c for c in df.columns]
    # income column may be 'income' or 'Income' etc.
    income_col = None
    for name in cols:
        if name.replace(".", "_").strip().lower() == "income":
            income_col = name
            break
    if income_col is None:
        raise ValueError("Target column 'income' not found.")
    return (df[income_col].astype(str).str.strip() == ">50K").astype(int)


def make_X_y_and_preprocessor(raw_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Full preparation:
      - initial clean/augment
      - y from income
      - ColumnTransformer fit
      - transform X to numpy array
    Returns: X, y, preprocessor, feature_names_out
    """
    # Run the cleaner first (does not remove 'income' column)
    cleaner = InitialCleaner()
    df_clean = cleaner.fit_transform(raw_df)

    # Produce y
    y = prepare_y(df_clean)

    # Drop original income column from features
    # After renaming, it will be 'income'
    drop_income_candidates = [c for c in df_clean.columns if c.replace(".", "_").strip().lower() == "income"]
    X_df = df_clean.drop(columns=drop_income_candidates, errors="ignore")

    # Build and fit preprocessor
    pre = build_preprocessor(X_df)
    X = pre.fit_transform(X_df)

    # Feature names
    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(X.shape[1])]

    return X, y.to_numpy(), pre, feat_names


# -----------------------------
# 3) Example usage
# -----------------------------
if __name__ == "__main__":
    # Example (assumes you've loaded the Adult dataset as a DataFrame `df`)
    # df = pd.read_csv("adult.data", header=None, names=[...])
    # Or from Kaggle API / local path
    # X, y, pre, names = make_X_y_and_preprocessor(df)
    pass