"""
Covariate shift detection: compares the distribution of live inference
inputs against the training distribution, per feature.

This is a model-health signal, separate from fraud flagging -- its output
feeds the covariate-shift alert tool, never the fraud-flag alert tool.
"""

from typing import Iterable

import pandas as pd
from scipy.stats import ks_2samp

from src.config import get_covariate_shift_alpha


def detect_covariate_shift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
    alpha: float | None = None,
) -> dict:
    """
    Run a two-sample Kolmogorov-Smirnov test per numeric feature, comparing
    the training-time reference distribution to a recent window of live
    inference inputs.

    Args:
        reference_df: Feature data from the training distribution.
        current_df: Feature data from a recent window of live inputs.
        feature_columns: Which columns to test. Defaults to all numeric
            columns common to both frames.
        alpha: Significance level below which a feature's p-value indicates
            drift. Falls back to `COVARIATE_SHIFT_ALPHA` (config).

    Returns:
        {
          "drifted": bool,                 # True if any tested feature drifted
          "drifted_features": [str, ...],  # features with p_value < alpha
          "per_feature": {feature: {"statistic": float, "p_value": float}},
        }
    """
    if alpha is None:
        alpha = get_covariate_shift_alpha()

    if feature_columns is None:
        numeric_ref = reference_df.select_dtypes(include="number").columns
        numeric_cur = current_df.select_dtypes(include="number").columns
        feature_columns = [c for c in numeric_ref if c in set(numeric_cur)]

    per_feature = {}
    drifted_features = []

    for feature in feature_columns:
        statistic, p_value = ks_2samp(
            reference_df[feature].dropna(),
            current_df[feature].dropna(),
        )
        per_feature[feature] = {"statistic": float(statistic), "p_value": float(p_value)}
        if p_value < alpha:
            drifted_features.append(feature)

    return {
        "drifted": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "per_feature": per_feature,
    }
