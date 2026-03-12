"""
AI Healthcare Data Platform – Feature Store.

Provides a lightweight, file-based feature store that supports:
- Versioned feature set registration and retrieval.
- Metadata tracking (schema, statistics, creation timestamp).
- Point-in-time correct feature lookup for training and serving.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_STORE_DIR = os.environ.get("FEATURE_STORE_DIR", "/data/feature_store")


class FeatureStore:
    """
    File-based feature store with versioning and metadata management.

    Each feature set is stored as a Parquet file under:
        <store_dir>/<feature_set_name>/v<version>/features.parquet
    Metadata is stored alongside as:
        <store_dir>/<feature_set_name>/v<version>/metadata.json
    """

    def __init__(self, store_dir: str = DEFAULT_STORE_DIR) -> None:
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        logger.info("Feature store initialised at: %s", store_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feature_set_path(self, name: str, version: int) -> str:
        return os.path.join(self.store_dir, name, f"v{version}")

    def _ensure_version_dir(self, name: str, version: int) -> str:
        path = self._feature_set_path(name, version)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        name: str,
        df: pd.DataFrame,
        version: int,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Persist a versioned feature set to the store.

        Args:
            name: Feature set name (e.g. 'patient_risk_features').
            df: DataFrame containing the features.
            version: Integer version number.
            description: Human-readable description of this feature set.
            tags: Optional list of tags for discoverability.

        Returns:
            Absolute path to the saved Parquet file.
        """
        version_dir = self._ensure_version_dir(name, version)
        parquet_path = os.path.join(version_dir, "features.parquet")
        metadata_path = os.path.join(version_dir, "metadata.json")

        df.to_parquet(parquet_path, index=False)

        metadata: Dict[str, Any] = {
            "name": name,
            "version": version,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "statistics": json.loads(df.describe(include="all").to_json()),
        }

        with open(metadata_path, "w") as fp:
            json.dump(metadata, fp, indent=2)

        logger.info(
            "Saved feature set '%s' v%d – %d rows, %d columns → %s",
            name,
            version,
            len(df),
            len(df.columns),
            parquet_path,
        )
        return parquet_path

    def load(self, name: str, version: int) -> pd.DataFrame:
        """
        Load a versioned feature set from the store.

        Args:
            name: Feature set name.
            version: Version to load.

        Returns:
            DataFrame of features.

        Raises:
            FileNotFoundError: If the specified version does not exist.
        """
        parquet_path = os.path.join(self._feature_set_path(name, version), "features.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Feature set '{name}' v{version} not found at {parquet_path}"
            )
        df = pd.read_parquet(parquet_path)
        logger.info("Loaded feature set '%s' v%d – %d rows", name, version, len(df))
        return df

    def get_metadata(self, name: str, version: int) -> Dict[str, Any]:
        """
        Retrieve metadata for a specific feature set version.

        Args:
            name: Feature set name.
            version: Version number.

        Returns:
            Metadata dictionary.
        """
        metadata_path = os.path.join(self._feature_set_path(name, version), "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata for '{name}' v{version} not found at {metadata_path}"
            )
        with open(metadata_path) as fp:
            return json.load(fp)

    def list_versions(self, name: str) -> List[int]:
        """
        List all available versions of a feature set.

        Args:
            name: Feature set name.

        Returns:
            Sorted list of version integers.
        """
        base_path = os.path.join(self.store_dir, name)
        if not os.path.exists(base_path):
            return []
        versions = []
        for entry in os.scandir(base_path):
            if entry.is_dir() and entry.name.startswith("v"):
                try:
                    versions.append(int(entry.name[1:]))
                except ValueError:
                    pass
        return sorted(versions)

    def latest_version(self, name: str) -> Optional[int]:
        """
        Return the latest version number for a feature set, or None if not found.

        Args:
            name: Feature set name.

        Returns:
            Latest version integer, or None.
        """
        versions = self.list_versions(name)
        return versions[-1] if versions else None
