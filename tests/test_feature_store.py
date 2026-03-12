"""Unit tests for the feature_store module."""
import os
import tempfile

import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

from feature_store import FeatureStore


@pytest.fixture
def tmp_store(tmp_path):
    return FeatureStore(store_dir=str(tmp_path))


@pytest.fixture
def sample_features():
    return pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [30, 45, 60],
            "risk_score": [0.1, 0.5, 0.9],
        }
    )


class TestFeatureStoreSave:
    def test_save_creates_parquet(self, tmp_store, sample_features):
        path = tmp_store.save("test_features", sample_features, version=1)
        assert os.path.exists(path)

    def test_save_creates_metadata(self, tmp_store, sample_features):
        tmp_store.save("test_features", sample_features, version=1, description="test")
        metadata = tmp_store.get_metadata("test_features", version=1)
        assert metadata["name"] == "test_features"
        assert metadata["version"] == 1
        assert metadata["num_rows"] == 3

    def test_save_multiple_versions(self, tmp_store, sample_features):
        tmp_store.save("feats", sample_features, version=1)
        tmp_store.save("feats", sample_features, version=2)
        assert tmp_store.list_versions("feats") == [1, 2]


class TestFeatureStoreLoad:
    def test_load_returns_dataframe(self, tmp_store, sample_features):
        tmp_store.save("feats", sample_features, version=1)
        loaded = tmp_store.load("feats", version=1)
        assert list(loaded.columns) == list(sample_features.columns)
        assert len(loaded) == len(sample_features)

    def test_load_missing_version_raises(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.load("nonexistent", version=99)


class TestFeatureStoreVersioning:
    def test_latest_version(self, tmp_store, sample_features):
        tmp_store.save("feats", sample_features, version=1)
        tmp_store.save("feats", sample_features, version=3)
        assert tmp_store.latest_version("feats") == 3

    def test_latest_version_none_if_empty(self, tmp_store):
        assert tmp_store.latest_version("no_such_feature") is None
