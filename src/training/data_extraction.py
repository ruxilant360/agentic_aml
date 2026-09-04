"""
Pulls training data out of PostgreSQL into a pandas DataFrame.

Training data lives in the `transactions` table in the DB (not a local CSV),
so every training run starts by extracting it here before any
preprocessing/modeling happens.
"""

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine

# Only rows with a known ground-truth label are usable for supervised training.
DEFAULT_TRAINING_QUERY = """
    SELECT
        transaction_id,
        step,
        type,
        amount,
        name_orig,
        oldbalance_org,
        newbalance_orig,
        name_dest,
        oldbalance_dest,
        newbalance_dest,
        is_flagged_fraud,
        is_fraud
    FROM transactions
    WHERE is_fraud IS NOT NULL
"""


def extract_training_data(query: str | None = None) -> pd.DataFrame:
    """
    Extract labeled transaction data from PostgreSQL as a pandas DataFrame.

    Args:
        query: Optional raw SQL to run instead of the default extraction
            query (e.g. to filter by date range or sample a subset). Must
            still return the columns expected by the preprocessing/training
            pipeline (see `DEFAULT_TRAINING_QUERY`).

    Returns:
        DataFrame with one row per labeled transaction.
    """
    sql = query if query is not None else DEFAULT_TRAINING_QUERY
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df
