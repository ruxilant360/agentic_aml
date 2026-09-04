"""
Node functions for the LangGraph agent graph.

Nodes own the *logic* (what to compute, which tool to call, how to fold the
result back into state); the tools in `src.tools` own the actual external
actions (DB write, RAG retrieval, alert delivery). Routing between nodes
lives in `src.agentic.graph`.
"""

from typing import Callable

from xgboost import XGBClassifier

from src.agentic.state import AgentState
from src.inference.shap_explainer import compute_shap_values
from src.tools.covariate_shift_alert_tool import raise_covariate_shift_alert
from src.tools.db_write_tool import write_transaction_record
from src.tools.fraud_alert_tool import raise_fraud_alert
from src.tools.report_tool import generate_policy_report


def make_explain_node(model: XGBClassifier) -> Callable[[AgentState], dict]:
    """
    Build the SHAP-explanation node, bound to a specific fitted model.

    The model is a factory argument (not module-level/global state) so the
    graph can be rebuilt against a different model artifact without editing
    this module.
    """

    def explain_node(state: AgentState) -> dict:
        shap_values = compute_shap_values(model, state["features"])
        return {"shap_values": shap_values}

    return explain_node


def write_db_node(state: AgentState) -> dict:
    """Always-run node: persists the transaction + model output regardless of flag."""
    result = write_transaction_record.invoke(
        {
            "transaction": state["transaction"],
            "score": state["score"],
            "flag": state["flag"],
            "shap_values": state["shap_values"],
        }
    )
    return {"log": state.get("log", []) + [result]}


def generate_report_node(state: AgentState) -> dict:
    """Only reached when `flag` is True (see graph routing)."""
    report = generate_policy_report.invoke(
        {
            "transaction_summary": state["transaction"],
            "shap_values": state["shap_values"],
        }
    )
    return {"report": report}


def fraud_alert_node(state: AgentState) -> dict:
    """Only reached when `flag` is True, after the report has been generated."""
    result = raise_fraud_alert.invoke(
        {
            "transaction_id": state["transaction"]["transaction_id"],
            "score": state["score"],
            "report": state["report"],
        }
    )
    return {"log": state.get("log", []) + [result]}


def covariate_shift_alert_node(state: AgentState) -> dict:
    """Covariate-shift path entry point -- independent of the transaction path."""
    result = raise_covariate_shift_alert.invoke({"drift_report": state["drift_report"]})
    return {"log": state.get("log", []) + [result]}
