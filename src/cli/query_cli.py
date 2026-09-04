"""
Terminal entrypoint for the LangChain NL-to-SQL query agent.

Not run as part of this change (per project convention: write the code,
don't execute it) -- invoke manually with `python -m src.cli.query_cli`.
"""

from src.query_agent.langchain_query_agent import build_query_agent


def main() -> None:
    agent = build_query_agent()
    print("AML database query assistant. Type a question, or 'exit' to quit.")

    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = agent.invoke({"input": user_input})
        print(result["output"])


if __name__ == "__main__":
    main()
