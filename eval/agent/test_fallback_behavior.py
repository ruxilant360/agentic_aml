"""
Fallback/no-op path tests for the LangGraph agent.

Verifies the agent's behavior when the fraud threshold is NOT reached:
the transaction (and its model output) must still be written to the
database, but no report should be generated and no fraud alert raised.
Also verifies the happy path (flagged) does trigger all three, as a
contrast/control.

Tools are stubbed out (no real DB/LLM calls) -- this tests the graph's
routing logic, not the tools' own implementations.
"""

import pandas as pd
import pytest

import src.agentic.nodes as nodes
from src.agentic.graph import build_agent_graph


class _StubTool:
    """Records calls made via `.invoke(...)`, mimicking a LangChain tool."""

    def __init__(self, return_value="ok"):
        self.calls = []
        self.return_value = return_value

    def invoke(self, payload):
        self.calls.append(payload)
        return self.return_value


@pytest.fixture
def stubbed_tools(monkeypatch):
    write_tool = _StubTool("wrote record")
    report_tool = _StubTool("drafted report")
    alert_tool = _StubTool("alert raised")

    monkeypatch.setattr(nodes, "write_transaction_record", write_tool)
    monkeypatch.setattr(nodes, "generate_policy_report", report_tool)
    monkeypatch.setattr(nodes, "raise_fraud_alert", alert_tool)
    monkeypatch.setattr(nodes, "compute_shap_values", lambda model, features: {"amount": 0.1})

    return {"write": write_tool, "report": report_tool, "alert": alert_tool}


def _sample_transaction_state(flag: bool) -> dict:
    return {
        "event_type": "transaction",
        "transaction": {
            "transaction_id": "T1",
            "step": 1,
            "type": "TRANSFER",
            "amount": 1000.0,
            "name_orig": "C1",
            "oldbalance_org": 5000.0,
            "newbalance_orig": 4000.0,
            "name_dest": "C2",
            "oldbalance_dest": 0.0,
            "newbalance_dest": 1000.0,
            "is_flagged_fraud": False,
        },
        "features": pd.DataFrame([{"amount": 1000.0}]),
        "score": 0.9 if flag else 0.1,
        "flag": flag,
    }


def test_no_op_path_when_not_flagged(stubbed_tools):
    graph = build_agent_graph(model=None)
    graph.invoke(_sample_transaction_state(flag=False))

    assert len(stubbed_tools["write"].calls) == 1, "DB write must still happen when not flagged"
    assert len(stubbed_tools["report"].calls) == 0, "no report should be generated when not flagged"
    assert len(stubbed_tools["alert"].calls) == 0, "no fraud alert should be raised when not flagged"


def test_write_call_has_correct_flag_value_on_no_op_path(stubbed_tools):
    graph = build_agent_graph(model=None)
    graph.invoke(_sample_transaction_state(flag=False))

    write_call = stubbed_tools["write"].calls[0]
    assert write_call["flag"] is False
    assert write_call["transaction"]["transaction_id"] == "T1"


def test_happy_path_when_flagged(stubbed_tools):
    graph = build_agent_graph(model=None)
    graph.invoke(_sample_transaction_state(flag=True))

    assert len(stubbed_tools["write"].calls) == 1
    assert len(stubbed_tools["report"].calls) == 1, "report must be generated when flagged"
    assert len(stubbed_tools["alert"].calls) == 1, "fraud alert must be raised when flagged"


def test_covariate_shift_path_does_not_touch_transaction_tools(stubbed_tools, monkeypatch):
    shift_alert_tool = _StubTool("shift alert raised")
    monkeypatch.setattr(nodes, "raise_covariate_shift_alert", shift_alert_tool)

    graph = build_agent_graph(model=None)
    graph.invoke({"event_type": "covariate_shift", "drift_report": {"drifted": True, "drifted_features": ["amount"]}})

    assert len(shift_alert_tool.calls) == 1
    assert len(stubbed_tools["write"].calls) == 0
    assert len(stubbed_tools["report"].calls) == 0
    assert len(stubbed_tools["alert"].calls) == 0
