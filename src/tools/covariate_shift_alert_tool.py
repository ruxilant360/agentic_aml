"""
LangGraph tool: raise a covariate-shift (model-health) alert.

Triggered by the inference pipeline's drift detection, not by any single
transaction's fraud score -- kept as its own tool, separate from
`raise_fraud_alert`, because the trigger, urgency, and downstream handling
(model/ops team vs. AML analyst) are different.
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger("aml.alerts.covariate_shift")


@tool
def raise_covariate_shift_alert(drift_report: dict) -> str:
    """
    Raise an alert notifying the ML/ops team that input feature
    distributions have drifted from the training distribution.

    Use this when `src.inference.covariate_shift.detect_covariate_shift`
    reports `drifted=True`. This is a model-health signal, not a
    transaction-level fraud flag -- do not use this tool to alert on an
    individual flagged transaction (use `raise_fraud_alert` for that).

    Args:
        drift_report: The dict returned by `detect_covariate_shift`
            (`drifted`, `drifted_features`, `per_feature` stats).

    Returns:
        A short confirmation string on success, or an error description
        string on failure (never raises).
    """
    try:
        logger.warning(
            "COVARIATE SHIFT ALERT: drifted_features=%s per_feature=%s",
            drift_report.get("drifted_features"),
            drift_report.get("per_feature"),
        )
        return f"Covariate shift alert raised for features: {drift_report.get('drifted_features')}."
    except Exception as exc:
        return f"Error raising covariate shift alert: {exc}"
