"""
ml/risk_prediction_model.py
============================
Patient risk prediction model for the AI Healthcare Data Platform.

Responsibilities
----------------
* Load cleansed clinical and (optionally) genomic staging data.
* Engineer features suitable for a gradient-boosted binary classifier.
* Train, cross-validate, and evaluate a scikit-learn GradientBoostingClassifier.
* Persist the trained model artefact with joblib for serving.
* Produce risk score predictions on new data and save them to Parquet.

Usage
-----
    # Train a new model
    python risk_prediction_model.py train \
        --clinical  data/staging/clinical \
        --output    models/risk_model.joblib

    # Score new patients
    python risk_prediction_model.py predict \
        --clinical  data/staging/clinical \
        --model     models/risk_model.joblib \
        --output    data/staging/risk_scores

    # Full train + predict in one shot
    python risk_prediction_model.py train-predict \
        --clinical  data/staging/clinical \
        --output    models/risk_model.joblib \
        --scores    data/staging/risk_scores
"""

import argparse
import glob
import logging
import os
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("risk-prediction")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_VERSION = "1.0.0"

# Feature columns expected after feature engineering
NUMERIC_FEATURES = [
    "age_at_registration",
    "num_encounters_90d",
    "avg_los_days",
    "num_distinct_diagnoses",
    "num_high_impact_variants",
]
CATEGORICAL_FEATURES = ["gender", "insurance_type", "primary_diagnosis_code"]

TARGET_COLUMN = "readmitted_30d"  # Binary label: 1 = readmitted within 30 days

RISK_THRESHOLDS = {
    "LOW": 0.0,
    "MEDIUM": 0.4,
    "HIGH": 0.7,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parquet_dir(directory: str) -> pd.DataFrame:
    """
    Read all Parquet files in *directory* into a single DataFrame.

    Parameters
    ----------
    directory : str
        Directory containing ``.parquet`` files (recursive search).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If no Parquet files are found in *directory*.
    """
    files = glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No Parquet files found in '{directory}'.")
    logger.info("Loading %d Parquet file(s) from '%s' …", len(files), directory)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    clinical_df: pd.DataFrame,
    genomic_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Derive a patient-level feature matrix from encounter and variant records.

    Clinical features (computed per patient)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``num_encounters_90d``     — encounters in the last 90-day window.
    * ``avg_los_days``           — mean length of stay across encounters.
    * ``num_distinct_diagnoses`` — count of unique ICD-10 codes.
    * ``primary_diagnosis_code`` — most frequent diagnosis code.

    Genomic features (if *genomic_df* is supplied)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``num_high_impact_variants`` — count of high-consequence variants.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Cleansed clinical records (one row per encounter).
    genomic_df : pd.DataFrame or None
        Cleaned genomic variant records.

    Returns
    -------
    pd.DataFrame
        One row per patient with engineered feature columns.
    """
    logger.info("Engineering features for %d clinical records …", len(clinical_df))

    clinical_df = clinical_df.copy()
    clinical_df["encounter_date"] = pd.to_datetime(
        clinical_df["encounter_date"], errors="coerce"
    )

    # Reference date for recency windows
    ref_date = clinical_df["encounter_date"].max()
    cutoff = ref_date - pd.Timedelta(days=90)

    # Per-patient aggregates
    agg = (
        clinical_df.groupby("patient_id")
        .agg(
            num_encounters_90d=(
                "encounter_date",
                lambda s: (s >= cutoff).sum(),
            ),
            avg_los_days=("length_of_stay_days", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            num_distinct_diagnoses=("diagnosis_code", "nunique"),
            primary_diagnosis_code=("diagnosis_code", lambda s: s.mode().iat[0] if len(s) else ""),
        )
        .reset_index()
    )

    # Readmission label (30-day window)
    clinical_df_sorted = clinical_df.sort_values(["patient_id", "encounter_date"])
    clinical_df_sorted["next_encounter"] = clinical_df_sorted.groupby("patient_id")[
        "encounter_date"
    ].shift(-1)
    clinical_df_sorted["days_to_next"] = (
        clinical_df_sorted["next_encounter"] - clinical_df_sorted["encounter_date"]
    ).dt.days
    readmit = (
        clinical_df_sorted.groupby("patient_id")["days_to_next"]
        .apply(lambda s: int((s <= 30).any()))
        .reset_index()
        .rename(columns={"days_to_next": TARGET_COLUMN})
    )

    features = agg.merge(readmit, on="patient_id", how="left")
    features[TARGET_COLUMN] = features[TARGET_COLUMN].fillna(0).astype(int)

    # Genomic features
    if genomic_df is not None and not genomic_df.empty:
        high_impact = (
            genomic_df[
                genomic_df["consequence"].isin(
                    [
                        "stop_gained",
                        "frameshift_variant",
                        "splice_acceptor_variant",
                        "splice_donor_variant",
                        "missense_variant",
                    ]
                )
            ]
            .groupby("sample_id")
            .size()
            .reset_index(name="num_high_impact_variants")
            .rename(columns={"sample_id": "patient_id"})
        )
        features = features.merge(high_impact, on="patient_id", how="left")
    features["num_high_impact_variants"] = (
        features.get("num_high_impact_variants", pd.Series(dtype=float))
        .fillna(0)
        .astype(int)
    )

    logger.info("Feature matrix shape: %s", features.shape)
    return features


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical feature columns in-place.

    Returns the modified DataFrame and a dict of fitted encoders keyed by
    column name (needed to apply the same encoding during prediction).
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna("unknown"))
        encoders[col] = le
    return df, encoders


def build_pipeline() -> Pipeline:
    """
    Construct a scikit-learn :class:`Pipeline` for the risk model.

    Architecture
    ~~~~~~~~~~~~
    1. ``StandardScaler`` — zero-mean, unit-variance scaling of numeric input.
    2. ``GradientBoostingClassifier`` — ensemble of 200 shallow decision trees.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    features: pd.DataFrame,
    model_output_path: str,
) -> dict[str, Any]:
    """
    Train the risk prediction model and save it to *model_output_path*.

    Parameters
    ----------
    features : pd.DataFrame
        Patient-level feature matrix including the target column
        :data:`TARGET_COLUMN`.
    model_output_path : str
        Path where the fitted model artefact will be saved (joblib format).

    Returns
    -------
    dict
        Training metrics: ``roc_auc``, ``cv_roc_auc_mean``,
        ``cv_roc_auc_std``, ``classification_report``.
    """
    all_feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available = [c for c in all_feature_cols if c in features.columns]

    encoded, encoders = _encode_categoricals(features[available + [TARGET_COLUMN]].copy())

    X = encoded[available].fillna(0)
    y = encoded[TARGET_COLUMN]

    logger.info(
        "Training on %d samples, %d features. Label distribution: %s",
        len(X),
        len(available),
        dict(y.value_counts()),
    )

    pipeline = build_pipeline()

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    logger.info(
        "CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std()
    )

    # Final fit on full dataset
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_proba)

    report = classification_report(y, y_pred)
    logger.info("Training ROC-AUC: %.4f", roc_auc)
    logger.info("Classification report:\n%s", report)

    # Save model artefact
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    artefact = {
        "pipeline": pipeline,
        "feature_columns": available,
        "encoders": encoders,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    joblib.dump(artefact, model_output_path)
    logger.info("Model saved → %s", model_output_path)

    return {
        "roc_auc": roc_auc,
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    features: pd.DataFrame,
    model_path: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate risk scores for a patient feature matrix.

    Parameters
    ----------
    features : pd.DataFrame
        Patient-level feature matrix (same schema as training, minus the label).
    model_path : str
        Path to the saved model artefact (joblib).
    output_dir : str
        Directory where the scored Parquet file will be written.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``patient_id``, ``risk_score``,
        ``risk_category``, ``scored_at``, ``model_version``.
    """
    artefact = joblib.load(model_path)
    pipeline: Pipeline = artefact["pipeline"]
    feature_columns: list[str] = artefact["feature_columns"]
    encoders: dict[str, LabelEncoder] = artefact["encoders"]
    model_version: str = artefact["model_version"]

    df = features.copy()
    for col, le in encoders.items():
        if col in df.columns:
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda v: v if v in known_classes else le.classes_[0]
            )
            df[col] = le.transform(df[col])
        else:
            df[col] = 0

    X = df.reindex(columns=feature_columns, fill_value=0).fillna(0)
    risk_scores = pipeline.predict_proba(X)[:, 1]

    def _categorise(score: float) -> str:
        if score >= RISK_THRESHOLDS["HIGH"]:
            return "HIGH"
        if score >= RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    scored_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    results = pd.DataFrame(
        {
            "patient_id": features["patient_id"].values,
            "risk_score": np.round(risk_scores, 6),
            "risk_category": [_categorise(s) for s in risk_scores],
            "scored_at": scored_at,
            "model_version": model_version,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(output_dir, f"risk_scores_{timestamp}.parquet")
    results.to_parquet(out_path, index=False)
    logger.info("Written %d risk scores → %s", len(results), out_path)

    # Summary log
    cat_counts = results["risk_category"].value_counts().to_dict()
    logger.info("Risk category distribution: %s", cat_counts)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Healthcare Platform — Patient Risk Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_p = subparsers.add_parser("train", help="Train and save a new model.")
    train_p.add_argument("--clinical", required=True, help="Clinical staging directory.")
    train_p.add_argument("--genomic", default=None, help="Genomic staging directory (optional).")
    train_p.add_argument("--output", default="models/risk_model.joblib", help="Model save path.")

    # --- predict ---
    pred_p = subparsers.add_parser("predict", help="Score patients with an existing model.")
    pred_p.add_argument("--clinical", required=True, help="Clinical staging directory.")
    pred_p.add_argument("--genomic", default=None, help="Genomic staging directory (optional).")
    pred_p.add_argument("--model", required=True, help="Path to saved model artefact.")
    pred_p.add_argument("--output", default="data/staging/risk_scores", help="Output directory.")

    # --- train-predict ---
    tp = subparsers.add_parser("train-predict", help="Train then immediately score.")
    tp.add_argument("--clinical", required=True, help="Clinical staging directory.")
    tp.add_argument("--genomic", default=None, help="Genomic staging directory (optional).")
    tp.add_argument("--output", default="models/risk_model.joblib", help="Model save path.")
    tp.add_argument("--scores", default="data/staging/risk_scores", help="Scores output directory.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    clinical_df = _load_parquet_dir(args.clinical)
    genomic_df = (
        _load_parquet_dir(args.genomic) if getattr(args, "genomic", None) else None
    )
    features = engineer_features(clinical_df, genomic_df)

    if args.command == "train":
        train(features, args.output)

    elif args.command == "predict":
        predict(features, args.model, args.output)

    elif args.command == "train-predict":
        train(features, args.output)
        predict(features, args.output, args.scores)


if __name__ == "__main__":
    main()
