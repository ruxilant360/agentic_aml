"""
SQL tool-calling accuracy: a fixed set of natural-language queries with
known-correct expectations, used to track how often the LangChain query
agent calls the DB tool correctly (right tool, query terms that should
show up in the generated SQL) vs. how often it gets it wrong.

Grading is keyword/pattern-based rather than exact-SQL-match, since the
underlying SQL is LLM-generated and can vary syntactically while still
being correct.
"""

from typing import Any, Callable

# Fixed set of (NL query, expectations) pairs. `expected_sql_keywords` are
# substrings (case-insensitive) that must appear in the SQL the tool
# generates for the query to count as correct.
SQL_TOOL_CALLING_CASES = [
    {
        "query": "How many transactions were flagged?",
        "expected_tool": "query_transactions_nl",
        "expected_sql_keywords": ["model_outputs", "flag"],
    },
    {
        "query": "Show me all TRANSFER type transactions.",
        "expected_tool": "query_transactions_nl",
        "expected_sql_keywords": ["transactions", "transfer"],
    },
    {
        "query": "What is the average fraud score?",
        "expected_tool": "query_transactions_nl",
        "expected_sql_keywords": ["avg", "score"],
    },
    {
        "query": "List the 5 highest-scored transactions.",
        "expected_tool": "query_transactions_nl",
        "expected_sql_keywords": ["order by", "score", "limit"],
    },
    {
        "query": "How many transactions came from name_orig 'C12345'?",
        "expected_tool": "query_transactions_nl",
        "expected_sql_keywords": ["name_orig", "c12345"],
    },
]


def score_tool_calling(
    invoke_and_capture: Callable[[str], dict[str, Any]],
    cases: list[dict] = SQL_TOOL_CALLING_CASES,
) -> dict:
    """
    Score tool-calling accuracy against a fixed set of NL queries.

    Args:
        invoke_and_capture: A callable taking a query string and returning
            {"tool_name": str, "generated_sql": str} describing which tool
            the agent called and (if applicable) the SQL it generated.
        cases: Test cases to grade against; defaults to
            `SQL_TOOL_CALLING_CASES`.

    Returns:
        {
          "accuracy": float,        # fraction of cases fully correct
          "results": [              # per-case detail
            {"query": str, "correct": bool, "reason": str | None}, ...
          ],
        }
    """
    results = []
    correct_count = 0

    for case in cases:
        capture = invoke_and_capture(case["query"])
        tool_name = capture.get("tool_name")
        generated_sql = (capture.get("generated_sql") or "").lower()

        if tool_name != case["expected_tool"]:
            results.append(
                {
                    "query": case["query"],
                    "correct": False,
                    "reason": f"called tool '{tool_name}', expected '{case['expected_tool']}'",
                }
            )
            continue

        missing = [kw for kw in case["expected_sql_keywords"] if kw.lower() not in generated_sql]
        if missing:
            results.append(
                {
                    "query": case["query"],
                    "correct": False,
                    "reason": f"generated SQL missing expected keywords: {missing}",
                }
            )
            continue

        results.append({"query": case["query"], "correct": True, "reason": None})
        correct_count += 1

    accuracy = correct_count / len(cases) if cases else 0.0
    return {"accuracy": accuracy, "results": results}


# ---- tests (exercise the scoring harness itself with a stub agent) ----

def test_score_tool_calling_all_correct():
    def stub_invoke(query: str) -> dict:
        case = next(c for c in SQL_TOOL_CALLING_CASES if c["query"] == query)
        fake_sql = "SELECT * FROM " + " ".join(case["expected_sql_keywords"])
        return {"tool_name": case["expected_tool"], "generated_sql": fake_sql}

    result = score_tool_calling(stub_invoke)
    assert result["accuracy"] == 1.0
    assert all(r["correct"] for r in result["results"])


def test_score_tool_calling_detects_wrong_tool():
    def stub_invoke(query: str) -> dict:
        return {"tool_name": "some_other_tool", "generated_sql": "SELECT 1"}

    result = score_tool_calling(stub_invoke)
    assert result["accuracy"] == 0.0
    assert all(not r["correct"] for r in result["results"])
    assert "called tool" in result["results"][0]["reason"]


def test_score_tool_calling_detects_missing_keywords():
    def stub_invoke(query: str) -> dict:
        case = next(c for c in SQL_TOOL_CALLING_CASES if c["query"] == query)
        return {"tool_name": case["expected_tool"], "generated_sql": "SELECT 1"}

    result = score_tool_calling(stub_invoke)
    assert result["accuracy"] == 0.0
    assert "missing expected keywords" in result["results"][0]["reason"]
