"""
LangGraph tool: raise an alert for a flagged (fraud-score) transaction.

Kept separate from the covariate-shift alert tool on purpose -- this fires
per-transaction, is triggered by the fraud flag, and is typically
higher-urgency/lower-volume than a model-health signal. Merging the two
would blur very different triggers and downstream handling.
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger("aml.alerts.fraud")


@tool
def raise_fraud_alert(transaction_id: str, score: float, report: str) -> str:
    """
    Raise an alert notifying analysts that a transaction was flagged.

    Use this only after `generate_policy_report` has produced the
    explanatory report for a flagged transaction. This tool delivers the
    alert (e.g. to a monitoring/notification channel); it does not itself
    decide whether the transaction is flagged.

    Args:
        transaction_id: Identifier of the flagged transaction.
        score: The model's fraud score for this transaction.
        report: The SHAP/policy-grounded report explaining the flag.

    Returns:
        A short confirmation string on success, or an error description
        string on failure (never raises).
    """
    try:
        # Alert delivery is a stand-in for whatever notification channel
        # (Slack, PagerDuty, email, etc.) this system integrates with --
        # swap the body of this function for that integration; the tool's
        # signature/docstring contract stays the same.
        logger.warning(
            "FRAUD ALERT: transaction_id=%s score=%.4f\n%s",
            transaction_id,
            score,
            report,
        )
        return f"Fraud alert raised for transaction {transaction_id}."
    except Exception as exc:
        return f"Error raising fraud alert for transaction {transaction_id}: {exc}"
