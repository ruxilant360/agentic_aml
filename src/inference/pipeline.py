"""
Inference pipeline: scores a transaction and returns a binary/score flag
only. This layer never labels a transaction "suspicious" (or similar) in
any user-facing text -- that framing is produced downstream by the
LangGraph agent via SHAP explanation.
"""

from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from src.config import get_fraud_threshold_default


def run_inference(
    transaction_features: pd.DataFrame,
    model: XGBClassifier,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Score a single (already-encoded) transaction and apply the flag threshold.

    Args:
        transaction_features: A single-row DataFrame of model-ready features
            (same encoding/columns used at training time).
        model: A fitted XGBoost classifier.
        threshold: Score threshold above which the transaction is flagged.
            Falls back to `FRAUD_SCORE_THRESHOLD` (config) if not supplied.
            Expected to be chosen by the precision-priority threshold
            selection process in `eval/model`, not an arbitrary 0.5 default.

    Returns:
        {"score": float, "flag": bool} -- a binary/score signal only, with
        no fraud-language label attached.
    """
    if threshold is None:
        threshold = get_fraud_threshold_default()

    score = float(model.predict_proba(transaction_features)[0, 1])
    flag = bool(score >= threshold)

    return {"score": score, "flag": flag}
