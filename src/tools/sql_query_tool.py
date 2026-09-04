"""
LangChain tool: natural-language query over the AML database.

This is the LangChain query agent's *only* tool -- read-only, terminal-
facing. Deliberately not shared with the LangGraph agent's tool set (that
agent writes/reports/alerts; this one only queries).
"""

import re

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from sqlalchemy import text

from src.config import get_groq_model_name
from src.db.connection import get_engine

_SCHEMA_DESCRIPTION = """\
Table transactions(
    transaction_id text, step int, type text, amount float,
    name_orig text, oldbalance_org float, newbalance_orig float,
    name_dest text, oldbalance_dest float, newbalance_dest float,
    is_flagged_fraud boolean, is_fraud boolean, created_at timestamp
)
Table model_outputs(
    transaction_id text, score float, flag boolean,
    shap_values jsonb, created_at timestamp
)
"""

_SQL_GEN_PROMPT = """\
You translate a natural-language question into a single read-only \
PostgreSQL SELECT query over this schema:

{schema}

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences.
- Only ever generate SELECT statements. Never write/modify/delete data.
- Use explicit column names, not SELECT *, where reasonable.

Question: {question}

SQL:
"""

_SELECT_ONLY_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def _nl_to_sql(question: str, llm: ChatGroq) -> str:
    prompt = _SQL_GEN_PROMPT.format(schema=_SCHEMA_DESCRIPTION, question=question)
    response = llm.invoke(prompt)
    sql = response.content.strip().strip("`").strip()
    # Some models wrap the query in a ```sql fence despite instructions; strip it.
    sql = re.sub(r"^sql\s*", "", sql, flags=re.IGNORECASE).strip()
    return sql


@tool
def query_transactions_nl(natural_language_query: str) -> str:
    """
    Answer a natural-language question about transactions/model outputs by
    translating it to SQL and querying the database.

    Read-only: only SELECT queries are ever executed against the database.
    Use this for questions like "how many transactions were flagged last
    week" or "show the top 5 highest-scored transactions".

    Args:
        natural_language_query: The user's question in plain English.

    Returns:
        A string representation of the query results, or an error
        description string on failure (never raises). Also returns an
        error string (without querying) if the generated SQL is not a
        read-only SELECT statement.
    """
    try:
        llm = ChatGroq(model=get_groq_model_name(), temperature=0)
        sql = _nl_to_sql(natural_language_query, llm)

        if not _SELECT_ONLY_RE.match(sql) or _FORBIDDEN_RE.search(sql):
            return (
                "Refused to execute generated query: only read-only SELECT "
                f"statements are permitted. Generated SQL was: {sql}"
            )

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            columns = list(result.keys())

        if not rows:
            return "Query returned no rows."

        formatted = [dict(zip(columns, row)) for row in rows]
        return str(formatted)
    except Exception as exc:
        return f"Error answering query: {exc}"
