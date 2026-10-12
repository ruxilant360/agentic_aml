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

    ser_id = Column(Integer, primary_key=True, autoincrement=True)
    transID = Column(String, unique=True, nullable=False, index=True)

    step = Column(Integer, nullable=False)
    type = Column(String, nullable=False)  # CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    amount = Column(Float, nullable=False)
    nameOrig = Column(String, nullable=False)
    oldbalanceOrg = Column(Float, nullable=False)
    newbalanceOrig = Column(Float, nullable=False)
    nameDest = Column(String, nullable=False)
    oldbalanceDest = Column(Float, nullable=False)
    newbalancDest = Column(Float, nullable=False)
    isFlaggedFraud = Column(Boolean, nullable=False, default=False)

    # Ground-truth label, when known (e.g. from historical/labeled data).
    # Nullable because live/unlabeled transactions won't have it at write time.
    isFraud = Column(Boolean, nullable=True)

    #created_at = Column(DateTime, server_default=func.now())

    model_output = relationship("ModelOutput", back_populates="transaction", uselist=False)


class ModelOutput(Base):
    """Model score/flag/explanation for one transaction. One row per scored transaction."""

    __tablename__ = "model_outputs"

    ser_id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, ForeignKey("transactions.transID"), nullable=False, index=True)

    score = Column(Float, nullable=False)
    flag = Column(Boolean, nullable=False)

    # Per-feature SHAP values, e.g. {"amount": 0.42, "oldbalance_org": -0.11, ...}.
    # Stored as JSON rather than normalized columns since the feature set can evolve.
    shap_values = Column(JSON, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    transaction = relationship("Transaction", back_populates="model_output")
