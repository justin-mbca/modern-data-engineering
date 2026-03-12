"""
AI Healthcare Data Platform – Health Data Preprocessing.

Provides modular preprocessing pipelines for clinical and genomic datasets:
- Missing-value imputation
- Outlier detection and capping
- Feature normalisation
- Categorical encoding
- Audit logging for every transformation step
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------


def impute_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Impute missing values for numeric and categorical columns.

    Args:
        df: Input DataFrame.
        numeric_strategy: 'mean' or 'median' for numeric columns.
        categorical_strategy: 'mode' or 'constant' for categorical columns.
        numeric_columns: Explicit list of numeric columns; inferred if None.
        categorical_columns: Explicit list of categorical columns; inferred if None.

    Returns:
        DataFrame with imputed values.
    """
    df = df.copy()
    num_cols = numeric_columns or df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = categorical_columns or df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in num_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            fill_value = df[col].median() if numeric_strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)
            logger.info("Imputed %d missing values in '%s' with %s=%.4f", missing, col, numeric_strategy, fill_value)

    for col in cat_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            mode_values = df[col].mode()
            if categorical_strategy == "mode" and len(mode_values) > 0:
                fill_value = mode_values[0]
            else:
                fill_value = "UNKNOWN"
            df[col] = df[col].fillna(fill_value)
            logger.info("Imputed %d missing values in '%s' with '%s'", missing, col, fill_value)

    return df


# ---------------------------------------------------------------------------
# Outlier detection and capping
# ---------------------------------------------------------------------------


def cap_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Cap outliers at IQR-based lower and upper fences.

    Args:
        df: Input DataFrame.
        columns: Columns to process; defaults to all numeric columns.
        iqr_multiplier: Multiplier for IQR fence calculation (default 1.5).

    Returns:
        Tuple of (capped DataFrame, dict mapping column → (lower_fence, upper_fence)).
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    fences: Dict[str, Tuple[float, float]] = {}

    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        fences[col] = (lower, upper)

        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped:
            logger.info("Capped %d outliers in '%s' to [%.4f, %.4f]", n_capped, col, lower, upper)

    return df, fences


# ---------------------------------------------------------------------------
# Feature normalisation
# ---------------------------------------------------------------------------


def normalise_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "standard",
) -> Tuple[pd.DataFrame, object]:
    """
    Normalise numeric features using standard or min-max scaling.

    Args:
        df: Input DataFrame.
        columns: Columns to scale; defaults to all numeric columns.
        method: 'standard' (z-score) or 'minmax' (0–1 range).

    Returns:
        Tuple of (scaled DataFrame, fitted scaler object).
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    df[cols] = scaler.fit_transform(df[cols])
    logger.info("Normalised %d columns using '%s' scaling: %s", len(cols), method, cols)
    return df, scaler


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def encode_categoricals(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns.

    Args:
        df: Input DataFrame.
        columns: Columns to encode; defaults to object/category columns.

    Returns:
        Tuple of (encoded DataFrame, dict mapping column → LabelEncoder).
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders: Dict[str, LabelEncoder] = {}

    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Label-encoded '%s' – %d unique classes", col, len(le.classes_))

    return df, encoders


# ---------------------------------------------------------------------------
# End-to-end preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_health_data(
    df: pd.DataFrame,
    not_null_columns: Optional[List[str]] = None,
    outlier_columns: Optional[List[str]] = None,
    normalise_columns: Optional[List[str]] = None,
    encode_columns: Optional[List[str]] = None,
    normalise_method: str = "standard",
) -> Dict:
    """
    Run the full preprocessing pipeline: impute → cap outliers → normalise → encode.

    Args:
        df: Raw health DataFrame.
        not_null_columns: Critical columns that must not contain nulls after imputation.
        outlier_columns: Numeric columns for outlier capping.
        normalise_columns: Numeric columns to normalise.
        encode_columns: Categorical columns to encode.
        normalise_method: 'standard' or 'minmax'.

    Returns:
        Dict with keys: 'dataframe', 'scaler', 'encoders', 'outlier_fences'.
    """
    logger.info("Starting health data preprocessing – input shape: %s", df.shape)

    df = impute_missing_values(df)

    df, fences = cap_outliers_iqr(df, columns=outlier_columns)

    df, scaler = normalise_features(df, columns=normalise_columns, method=normalise_method)

    df, encoders = encode_categoricals(df, columns=encode_columns)

    # Final null check on critical columns
    if not_null_columns:
        nulls = {col: int(df[col].isna().sum()) for col in not_null_columns if col in df.columns}
        failing = {col: cnt for col, cnt in nulls.items() if cnt > 0}
        if failing:
            raise ValueError(f"Null values remain in critical columns after preprocessing: {failing}")

    logger.info("Preprocessing complete – output shape: %s", df.shape)
    return {
        "dataframe": df,
        "scaler": scaler,
        "encoders": encoders,
        "outlier_fences": fences,
    }
