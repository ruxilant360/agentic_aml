"""
Golden evals: a fixed set of representative user prompts against the
LangChain query agent, with per-prompt response time measured -- used to
track latency regressions over time, not correctness.
"""

import time
from typing import Callable

# Representative NL questions a terminal user is expected to ask. Kept
# small and fixed so latency numbers are comparable run over run.
GOLDEN_PROMPTS = [
    "How many transactions were flagged in the last 24 hours?",
    "What is the average fraud score for CASH_OUT transactions?",
    "Show the 5 highest-scored transactions.",
    "How many transactions of type TRANSFER have flag = true?",
    "What is the total amount moved by flagged transactions this week?",
]


def run_golden_evals(agent_invoke: Callable[[str], str]) -> list[dict]:
    """
    Run each golden prompt through the given agent-invocation callable and
    record wall-clock latency.

    Args:
        agent_invoke: A callable taking a single prompt string and
            returning the agent's response string (typically
            `lambda q: build_query_agent().invoke({"input": q})["output"]`,
            or a pre-built agent's equivalent wrapper).

    Returns:
        A list of {"prompt": str, "response": str, "latency_seconds": float}
        dicts, one per golden prompt, in `GOLDEN_PROMPTS` order.
    """
    results = []
    for prompt in GOLDEN_PROMPTS:
        start = time.perf_counter()
        response = agent_invoke(prompt)
        elapsed = time.perf_counter() - start
        results.append({"prompt": prompt, "response": response, "latency_seconds": elapsed})
    return results


def check_latency_regression(
    results: list[dict],
    max_latency_seconds: float,
) -> list[str]:
    """
    Flag any golden-eval result whose latency exceeds the given budget.

    Args:
        results: Output of `run_golden_evals`.
        max_latency_seconds: Latency budget per prompt.

    Returns:
        Prompts (strings) that exceeded the budget. Empty list means no
        regressions detected.
    """
    return [r["prompt"] for r in results if r["latency_seconds"] > max_latency_seconds]
