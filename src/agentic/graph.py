"""
LangGraph agent graph definition.

One agent, two entry triggers (`event_type`):

  "transaction"      explain (SHAP) -> write to DB -> [if flagged] generate
                      report -> raise fraud alert -> END
                                 [if not flagged]  -> END (no report, no
                      alert -- the fallback/no-op path exercised by
                      eval/agent/test_fallback_behavior.py)

  "covariate_shift"   raise covariate-shift alert -> END
                      (separate from the fraud-flag path entirely: no DB
                      write, no report -- this is a model-health signal)

Routing logic (which branch to take) lives here; the nodes themselves stay
single-purpose (see `src.agentic.nodes`).
"""

from langgraph.graph import StateGraph, START, END
from xgboost import XGBClassifier

from src.agentic.nodes import (
    covariate_shift_alert_node,
    fraud_alert_node,
    generate_report_node,
    make_explain_node,
    write_db_node,
)
from src.agentic.state import AgentState


def _route_event(state: AgentState) -> str:
    return state["event_type"]


def _route_after_write(state: AgentState) -> str:
    return "flagged" if state.get("flag") else "no_op"


def build_agent_graph(model: XGBClassifier):
    """
    Construct and compile the LangGraph agent.

    Args:
        model: The fitted XGBoost classifier used to compute SHAP
            explanations for flagged/scored transactions.

    Returns:
        A compiled LangGraph graph, invokable via `.invoke(state)` /
        `.stream(state)` with an `AgentState` (must set `event_type` plus
        either the transaction-path or covariate-shift-path fields).
    """
    explain_node = make_explain_node(model)

    graph = StateGraph(AgentState)

    graph.add_node("explain", explain_node)
    graph.add_node("write_db", write_db_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("fraud_alert", fraud_alert_node)
    graph.add_node("covariate_shift_alert", covariate_shift_alert_node)

    graph.add_conditional_edges(
        START,
        _route_event,
        {
            "transaction": "explain",
            "covariate_shift": "covariate_shift_alert",
        },
    )

    graph.add_edge("explain", "write_db")

    graph.add_conditional_edges(
        "write_db",
        _route_after_write,
        {
            "flagged": "generate_report",
            "no_op": END,
        },
    )

    graph.add_edge("generate_report", "fraud_alert")
    graph.add_edge("fraud_alert", END)
    graph.add_edge("covariate_shift_alert", END)

    return graph.compile()
