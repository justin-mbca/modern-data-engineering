"""
AI Healthcare Data Platform – Risk Classification Model with Explainability.

Trains a gradient-boosted classifier to predict patient health risk levels
(Low / Medium / High) and provides global and local explanations using SHAP.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_risk_model(
    df: pd.DataFrame,
    target_column: str = "risk_level",
    feature_columns: Optional[list] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[GradientBoostingClassifier, pd.DataFrame, pd.Series, dict]:
    """
    Train a GradientBoostingClassifier for patient risk classification.

    Args:
        df: Pre-processed DataFrame with features and target.
        target_column: Name of the target column (integer-encoded risk level).
        feature_columns: List of feature column names; inferred if None.
        test_size: Proportion of data reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (trained model, X_test DataFrame, y_test Series, metrics dict).
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "Training on %d samples, evaluating on %d samples. Features: %d",
        len(X_train),
        len(X_test),
        len(feature_columns),
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "classification_report": classification_report(
            y_test, y_pred, target_names=list(RISK_LABELS.values()), output_dict=True
        ),
    }
    try:
        if len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            metrics["roc_auc_ovr"] = roc_auc_score(y_test, y_proba, multi_class="ovr")
    except ValueError:
        pass

    logger.info("Model training complete.\n%s", classification_report(y_test, y_pred))
    return model, X_test, y_test, metrics


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------


def explain_model_global(
    model: GradientBoostingClassifier,
    X: pd.DataFrame,
    max_display: int = 20,
    output_dir: Optional[str] = None,
) -> shap.Explainer:
    """
    Compute global SHAP feature importances and optionally save a summary plot.

    Args:
        model: Trained GradientBoostingClassifier.
        X: Feature DataFrame (used as background for the explainer).
        max_display: Number of top features to show in the summary plot.
        output_dir: Directory for saving the summary plot PNG; skipped if None.

    Returns:
        Fitted shap.Explainer instance.
    """
    logger.info("Computing global SHAP values for %d samples…", len(X))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    logger.info("Global SHAP computation complete.")

    if output_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plot_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved to %s", plot_path)

    return explainer


def explain_instance(
    explainer: shap.Explainer,
    instance: pd.DataFrame,
    class_index: int = 2,
) -> Dict:
    """
    Generate a local SHAP explanation for a single patient instance.

    Args:
        explainer: Pre-fitted SHAP TreeExplainer.
        instance: Single-row DataFrame for the patient to explain.
        class_index: Class index to explain (default 2 = 'High' risk).

    Returns:
        Dict with 'base_value', 'shap_values', and 'feature_contributions'.
    """
    shap_values = explainer.shap_values(instance)
    # shap_values is list[class_idx] of arrays when multi-class
    if isinstance(shap_values, list):
        sv = shap_values[class_index][0]
        base = explainer.expected_value[class_index]
    else:
        sv = shap_values[0]
        base = explainer.expected_value

    contributions = dict(zip(instance.columns, sv))
    top_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    logger.info(
        "Local SHAP explanation for class '%s': top features = %s",
        RISK_LABELS.get(class_index, class_index),
        top_features[:3],
    )

    return {
        "base_value": float(base),
        "shap_values": sv.tolist(),
        "feature_contributions": dict(top_features),
    }


# ---------------------------------------------------------------------------
# End-to-end risk classification pipeline
# ---------------------------------------------------------------------------


def run_risk_classification_pipeline(
    df: pd.DataFrame,
    target_column: str = "risk_level",
    feature_columns: Optional[list] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Train the risk model and generate global + local SHAP explanations.

    Args:
        df: Pre-processed patient DataFrame.
        target_column: Target column name.
        feature_columns: Feature columns; inferred if None.
        output_dir: Directory for saving SHAP plots.

    Returns:
        Dict with 'model', 'metrics', 'explainer', and 'sample_explanation'.
    """
    model, X_test, y_test, metrics = train_risk_model(
        df, target_column=target_column, feature_columns=feature_columns
    )

    explainer = explain_model_global(model, X_test, output_dir=output_dir)

    sample = X_test.iloc[[0]]
    local_explanation = explain_instance(explainer, sample)

    return {
        "model": model,
        "metrics": metrics,
        "explainer": explainer,
        "sample_explanation": local_explanation,
    }
