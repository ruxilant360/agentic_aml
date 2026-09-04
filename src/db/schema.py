"""
ORM mapping onto the two pre-existing PostgreSQL tables.

These classes describe the schema the tables are assumed to already have --
they are read/write mappings, not creation DDL. If the live schema differs,
adjust column definitions here; nothing in this file creates or alters
tables.

    transactions   -- raw transaction data + ground-truth label (if known)
    model_outputs  -- one row per scored transaction: score, flag, SHAP
                       values (per feature), tied back to its transaction
"""

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Transaction(Base):
    """A single transaction record, mirroring the source dataset's columns."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, unique=True, nullable=False, index=True)

    step = Column(Integer, nullable=False)
    type = Column(String, nullable=False)  # CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    amount = Column(Float, nullable=False)
    name_orig = Column(String, nullable=False)
    oldbalance_org = Column(Float, nullable=False)
    newbalance_orig = Column(Float, nullable=False)
    name_dest = Column(String, nullable=False)
    oldbalance_dest = Column(Float, nullable=False)
    newbalance_dest = Column(Float, nullable=False)
    is_flagged_fraud = Column(Boolean, nullable=False, default=False)

    # Ground-truth label, when known (e.g. from historical/labeled data).
    # Nullable because live/unlabeled transactions won't have it at write time.
    is_fraud = Column(Boolean, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    model_output = relationship("ModelOutput", back_populates="transaction", uselist=False)


class ModelOutput(Base):
    """Model score/flag/explanation for one transaction. One row per scored transaction."""

    __tablename__ = "model_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"), nullable=False, index=True)

    score = Column(Float, nullable=False)
    flag = Column(Boolean, nullable=False)

    # Per-feature SHAP values, e.g. {"amount": 0.42, "oldbalance_org": -0.11, ...}.
    # Stored as JSON rather than normalized columns since the feature set can evolve.
    shap_values = Column(JSON, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    transaction = relationship("Transaction", back_populates="model_output")
