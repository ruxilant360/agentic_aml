"""
SHAP-based per-feature explanations for a scored transaction.

Computed once per transaction and shared by the agent: stored verbatim in
the `model_outputs` table, and consumed by the RAG report tool to explain
*why* a transaction was flagged. This module only computes numeric SHAP
values -- it never generates fraud-language text itself.
"""

import pandas as pd
import shap
from xgboost import XGBClassifier


def compute_shap_values(
    model: XGBClassifier,
    transaction_features: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute per-feature SHAP values for a single transaction.

    Args:
        model: A fitted XGBoost classifier.
        transaction_features: Single-row DataFrame of model-ready features,
            same columns/encoding used at training time.

    Returns:
        Mapping of feature name -> SHAP value for that transaction, e.g.
        {"amount": 0.42, "oldbalance_org": -0.11, ...}. Sign/magnitude
        indicate each feature's contribution to the model's score, not a
        verbal judgment -- turning this into explanatory text is the RAG
        report tool's job, not this function's.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transaction_features)

    # TreeExplainer returns a 2D array (n_samples, n_features) for binary
    # classification with a single-output model; take the one row we have.
    row = shap_values[0] if getattr(shap_values, "ndim", 1) == 2 else shap_values

    return dict(zip(transaction_features.columns, [float(v) for v in row]))


def top_contributing_features(
    shap_values: dict[str, float],
    n: int = 5,
) -> list[tuple[str, float]]:
    """
    Return the `n` features with the largest absolute SHAP contribution,
    ordered most-influential first. Used to keep report generation focused
    on what actually drove the score rather than dumping every feature.
    """
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ranked[:n]
