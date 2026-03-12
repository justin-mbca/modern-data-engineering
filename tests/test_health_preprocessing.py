"""Unit tests for the health_preprocessing module."""
import numpy as np
import pandas as pd
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_ingestion"))

from health_preprocessing import (
    cap_outliers_iqr,
    encode_categoricals,
    impute_missing_values,
    normalise_features,
    preprocess_health_data,
)


@pytest.fixture
def sample_df():
    np.random.seed(0)
    return pd.DataFrame(
        {
            "age": [25, 45, np.nan, 60, 35],
            "bmi": [22.0, 30.5, 27.1, np.nan, 24.3],
            "diagnosis": ["A", "B", np.nan, "A", "C"],
            "gender": ["M", "F", "M", "F", "M"],
        }
    )


class TestImputation:
    def test_imputes_numeric_median(self, sample_df):
        result = impute_missing_values(sample_df, numeric_strategy="median")
        assert result["age"].isna().sum() == 0
        assert result["bmi"].isna().sum() == 0

    def test_imputes_categorical_mode(self, sample_df):
        result = impute_missing_values(sample_df, categorical_strategy="mode")
        assert result["diagnosis"].isna().sum() == 0

    def test_no_data_loss_on_full_df(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        result = impute_missing_values(df)
        assert result.shape == df.shape


class TestOutlierCapping:
    def test_caps_extreme_values(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        capped, fences = cap_outliers_iqr(df)
        assert capped["val"].max() < 100
        assert "val" in fences

    def test_returns_fence_dict(self):
        df = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
        _, fences = cap_outliers_iqr(df)
        lower, upper = fences["a"]
        assert lower < upper


class TestNormalisation:
    def test_standard_scaling_mean_zero(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaled, scaler = normalise_features(df, method="standard")
        assert abs(scaled["x"].mean()) < 1e-10

    def test_minmax_scaling_range(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
        scaled, _ = normalise_features(df, method="minmax")
        assert scaled["x"].min() >= 0.0
        assert scaled["x"].max() <= 1.0


class TestEncoding:
    def test_encode_categoricals_no_nulls(self, sample_df):
        df = impute_missing_values(sample_df)
        encoded, encoders = encode_categoricals(df)
        assert encoded["gender"].dtype in [np.int32, np.int64, "int32", "int64"]
        assert "gender" in encoders

    def test_encoder_maps_correctly(self):
        df = pd.DataFrame({"cat": ["low", "medium", "high", "low"]})
        encoded, encoders = encode_categoricals(df)
        le = encoders["cat"]
        assert set(le.classes_) == {"high", "low", "medium"}


class TestPipeline:
    def test_full_pipeline_runs(self, sample_df):
        result = preprocess_health_data(sample_df)
        assert "dataframe" in result
        assert result["dataframe"].shape[0] == len(sample_df)

    def test_pipeline_raises_on_critical_null(self):
        # Use a pandas nullable Float64 column (not selected by np.number select_dtypes),
        # so imputation leaves it untouched and the null check fires.
        df = pd.DataFrame(
            {
                "id": pd.array([pd.NA, pd.NA, pd.NA], dtype=pd.Float64Dtype()),
                "val": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="Null values remain"):
            preprocess_health_data(
                df,
                not_null_columns=["id"],
                normalise_columns=["val"],
                encode_columns=[],
            )
