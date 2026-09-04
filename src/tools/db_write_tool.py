"""
LangGraph tool: persist a scored transaction to PostgreSQL.

Writes to both tables: the raw transaction row (`transactions`) and its
model output row (`model_outputs`) -- score, flag, per-feature SHAP values.
This is the one tool responsible for all DB writes; report generation and
alerting tools never write to the DB themselves.
"""

from langchain_core.tools import tool

from src.db.connection import get_session
from src.db.schema import Transaction, ModelOutput


@tool
def write_transaction_record(
    transaction: dict,
    score: float,
    flag: bool,
    shap_values: dict,
) -> str:
    """
    Persist a transaction and its model output to the database.

    Use this after a transaction has been scored, regardless of whether it
    was flagged -- every scored transaction gets a `transactions` row and a
    matching `model_outputs` row (score, flag, SHAP values), so downstream
    NL-to-SQL queries can find it either way.

    Args:
        transaction: Raw transaction fields. Must include `transaction_id`,
            `step`, `type`, `amount`, `name_orig`, `oldbalance_org`,
            `newbalance_orig`, `name_dest`, `oldbalance_dest`,
            `newbalance_dest`, `is_flagged_fraud`, and optionally
            `is_fraud` (ground truth, if known).
        score: Model's fraud score for this transaction (0-1).
        flag: Whether the score crossed the decision threshold.
        shap_values: Per-feature SHAP values explaining the score.

    Returns:
        A short confirmation string on success, or an error description
        string on failure (never raises).
    """
    try:
        with get_session() as session:
            txn_row = Transaction(
                transaction_id=transaction["transaction_id"],
                step=transaction["step"],
                type=transaction["type"],
                amount=transaction["amount"],
                name_orig=transaction["name_orig"],
                oldbalance_org=transaction["oldbalance_org"],
                newbalance_orig=transaction["newbalance_orig"],
                name_dest=transaction["name_dest"],
                oldbalance_dest=transaction["oldbalance_dest"],
                newbalance_dest=transaction["newbalance_dest"],
                is_flagged_fraud=transaction.get("is_flagged_fraud", False),
                is_fraud=transaction.get("is_fraud"),
            )
            output_row = ModelOutput(
                transaction_id=transaction["transaction_id"],
                score=score,
                flag=flag,
                shap_values=shap_values,
            )
            session.merge(txn_row)
            session.add(output_row)
            session.commit()

        return f"Wrote transaction {transaction['transaction_id']} (flag={flag}, score={score:.4f}) to the database."
    except Exception as exc:
        return f"Error writing transaction record to the database: {exc}"
