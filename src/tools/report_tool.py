"""
LangGraph tool: generate a policy-grounded, SHAP-explained report for a
flagged transaction (RAG). Only meant to be invoked once a transaction has
already been flagged -- see `src.agentic.graph` for the routing that
decides when this tool is reachable.
"""

from langchain_core.tools import tool

from src.rag.report_generator import generate_report
from src.rag.vector_store import build_policy_vector_store

# Built lazily and cached at module scope: embedding + FAISS index
# construction is expensive and the policy corpus doesn't change per call.
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_policy_vector_store()
    return _vectorstore


@tool
def generate_policy_report(transaction_summary: dict, shap_values: dict) -> str:
    """
    Draft a policy-grounded report explaining why a transaction was flagged.

    Retrieves relevant AML policy passages and combines them with the
    transaction's SHAP feature contributions to explain the reasoning
    behind the flag. The report never states a definitive fraud verdict
    (e.g. never says "suspicious") -- it surfaces the SHAP-driven reasoning
    for an analyst to interpret.

    Args:
        transaction_summary: Human-readable transaction fields (amount,
            type, step, etc.) for context.
        shap_values: Per-feature SHAP values for the flagged transaction.

    Returns:
        The drafted report text, or an error description string on failure
        (never raises).
    """
    try:
        vectorstore = _get_vectorstore()
        return generate_report(
            transaction_summary=transaction_summary,
            shap_values=shap_values,
            vectorstore=vectorstore,
        )
    except Exception as exc:
        return f"Error generating policy report: {exc}"
