"""
Preprocessing for the (already-cleaned) transaction dataset: categorical
encoding and the train/test split. No imputation/outlier handling here --
the source data is assumed clean per project convention.
"""

from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# High-cardinality identifiers -- not useful as model features, and not
# categorical in any meaningful sense (each value is essentially unique).
ID_COLUMNS = ["transaction_id", "name_orig", "name_dest"]

# `is_flagged_fraud` is an existing rule-based flag on the same data the
# model is trying to predict over -- treating it as a feature risks leaking
# the very signal we're trying to independently model. Excluded from
# features; kept available upstream for analysis/comparison only.
LEAKAGE_COLUMNS = ["is_flagged_fraud"]

TARGET_COLUMN = "is_fraud"

CATEGORICAL_COLUMNS = ["type"]


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: Iterable[str] = CATEGORICAL_COLUMNS,
    encoder: OneHotEncoder | None = None,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode categorical columns (e.g. `type`).

    Args:
        df: Input DataFrame containing the raw categorical columns.
        categorical_columns: Names of columns to one-hot encode.
        encoder: A previously-fit OneHotEncoder to reuse at inference time
            (transform-only). If None, a new encoder is fit on `df`
            (training time).

    Returns:
        (encoded_df, encoder) -- encoded_df has the categorical columns
        replaced by their one-hot columns; encoder is the fitted
        OneHotEncoder, to be persisted alongside the model and reused for
        inference so train/inference encodings stay consistent.
    """
    categorical_columns = list(categorical_columns)
    df = df.copy()

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(df[categorical_columns])
    else:
        encoded = encoder.transform(df[categorical_columns])

    encoded_cols = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder


def split_train_test(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
    leakage_columns: Iterable[str] = LEAKAGE_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train/test feature matrices and label vectors.

    Drops identifier columns and known leakage columns from the feature
    matrix before splitting; the target column is separated out as `y`.

    Args:
        df: Full labeled dataset (post categorical encoding).
        target_column: Name of the label column.
        id_columns: Identifier columns to exclude from features.
        leakage_columns: Columns excluded from features due to leakage risk.
        test_size: Fraction of rows held out for testing.
        random_state: Seed for reproducibility.
        stratify: Whether to stratify the split on the target (recommended
            given class imbalance in fraud data).

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    drop_cols = [c for c in (*id_columns, *leakage_columns) if c in df.columns]

    y = df[target_column]
    X = df.drop(columns=[target_column, *drop_cols])

    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return X_train, X_test, y_train, y_test
