"""
Shared state schema for the LangGraph agent.

A single graph handles two distinct triggers -- a newly scored transaction,
and a covariate-shift signal from the inference pipeline -- selected via
`event_type`. Fields not relevant to a given event type are simply left
unset for that run.
"""

from typing import Literal, NotRequired, TypedDict


class AgentState(TypedDict):
    event_type: Literal["transaction", "covariate_shift"]

    # -- transaction path --
    # `transaction`: raw fields (for DB write / human-readable report context).
    # `features`: model-ready encoded feature row used only for SHAP computation.
    # `score`/`flag`: already produced upstream by the inference pipeline --
    # this agent reacts to them, it does not (re)score the transaction.
    transaction: NotRequired[dict]
    features: NotRequired[object]  # pandas.DataFrame, single row
    score: NotRequired[float]
    flag: NotRequired[bool]
    shap_values: NotRequired[dict]
    report: NotRequired[str]

    # -- covariate-shift path --
    drift_report: NotRequired[dict]

    # -- shared --
    log: NotRequired[list[str]]
