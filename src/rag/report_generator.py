"""
Drafts a policy-grounded report explaining *why* a transaction was flagged,
using SHAP values to drive the explanation. The generated text must never
assert the literal word "suspicious" (or an equivalent definitive fraud
label) -- it surfaces SHAP-driven reasoning and policy context, and leaves
the "is this suspicious" judgment to the analyst reading it.
"""

from langchain_core.vectorstores import VectorStore
from langchain_groq import ChatGroq

from src.config import get_groq_model_name
from src.inference.shap_explainer import top_contributing_features

# The model is instructed never to use fraud-verdict language itself; the
# report describes contributing factors, not a conclusion.
_REPORT_SYSTEM_PROMPT = """\
You are an AML analyst-support assistant. You write short factual reports \
that explain why a transaction was flagged by a model, using the provided \
SHAP feature contributions and relevant policy excerpts.

Rules:
- Do NOT use the word "suspicious" or any equivalent definitive fraud \
  verdict (e.g. "this is fraud", "confirmed money laundering"). Describe \
  contributing factors and cite policy; let the analyst draw conclusions.
- Ground every claim in either the SHAP values given or the policy context \
  given. Do not invent facts not present in either.
- Be concise: a short paragraph, not a long essay.
"""


def _build_prompt(transaction_summary: dict, shap_values: dict, policy_context: str) -> str:
    top_features = top_contributing_features(shap_values, n=5)
    features_text = "\n".join(f"- {name}: {value:+.4f}" for name, value in top_features)

    return f"""\
Transaction summary:
{transaction_summary}

Top contributing model features (SHAP values, signed by direction of contribution):
{features_text}

Relevant policy context:
{policy_context}

Write the report now.
"""


def generate_report(
    transaction_summary: dict,
    shap_values: dict,
    vectorstore: VectorStore,
    llm: ChatGroq | None = None,
    k: int = 3,
) -> str:
    """
    Generate a SHAP-grounded, policy-cited report for a flagged transaction.

    Args:
        transaction_summary: A small dict of human-readable transaction
            fields (e.g. amount, type, step) to give the LLM context.
        shap_values: Per-feature SHAP values for this transaction, as
            returned by `src.inference.shap_explainer.compute_shap_values`.
        vectorstore: A FAISS (or other) vector store built over policy docs
            (see `src.rag.vector_store.build_policy_vector_store`).
        llm: Optional pre-built ChatGroq instance (mainly for testing);
            defaults to constructing one from config.
        k: Number of policy chunks to retrieve.

    Returns:
        The drafted report text.
    """
    if llm is None:
        llm = ChatGroq(model=get_groq_model_name(), temperature=0)

    top_features = top_contributing_features(shap_values, n=5)
    retrieval_query = "policy guidance relevant to: " + ", ".join(
        name for name, _ in top_features
    )
    docs = vectorstore.similarity_search(retrieval_query, k=k)
    policy_context = "\n\n".join(doc.page_content for doc in docs)

    prompt = _build_prompt(transaction_summary, shap_values, policy_context)

    response = llm.invoke(
        [
            {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    return response.content
