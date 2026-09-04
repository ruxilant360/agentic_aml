"""
LangChain agent that lets a user query the AML database via natural
language from the terminal. Read-only, single-tool -- intentionally kept
separate from the LangGraph agent's write/report/alert tool set.
"""

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import get_groq_model_name
from src.tools.sql_query_tool import query_transactions_nl

_SYSTEM_PROMPT = """\
You are a read-only assistant for querying an AML transaction database. \
Use the query_transactions_nl tool to answer the user's question. Do not \
attempt to write, update, or delete any data -- you have no tool for that. \
Answer concisely based only on what the tool returns.
"""


def build_query_agent() -> AgentExecutor:
    """
    Build the LangChain NL-to-SQL query agent.

    Returns:
        An `AgentExecutor` with a single tool (`query_transactions_nl`),
        ready to be invoked with {"input": "<natural language question>"}.
    """
    llm = ChatGroq(model=get_groq_model_name(), temperature=0)
    tools = [query_transactions_nl]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)
