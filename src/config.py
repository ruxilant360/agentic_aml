"""
Central configuration loader.

All environment-dependent values (DB connection, API keys, file paths) are
read here so the rest of the codebase never calls `os.environ` / `os.getenv`
directly. Values are read lazily via functions (not module-level constants)
so importing this module never fails just because an env var is missing --
the failure happens only when the specific value is actually needed.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:
    """
    Build the PostgreSQL connection URL used by SQLAlchemy.

    Prefers a single DATABASE_URL env var; falls back to assembling one from
    discrete PG_* env vars so local dev doesn't require hand-building a URL.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE", "aml")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment (.env).")
    return key


def get_groq_model_name() -> str:
    return os.getenv("GROQ_MODEL_NAME", "qwen/qwen3.6-27b")


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


def get_policy_docs_dir() -> str:
    """Directory containing the policy PDFs/text docs used by the RAG report tool."""
    return os.getenv("POLICY_DOCS_DIR", "policy_docs")


def get_fraud_threshold_default() -> float:
    """Default score threshold used at inference time if none is supplied explicitly."""
    return float(os.getenv("FRAUD_SCORE_THRESHOLD", "0.5"))


def get_covariate_shift_alpha() -> float:
    """Significance level for the covariate-shift statistical test."""
    return float(os.getenv("COVARIATE_SHIFT_ALPHA", "0.05"))
