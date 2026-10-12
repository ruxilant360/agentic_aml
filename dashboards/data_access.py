"""
Data access for the AML dashboards.

Two interchangeable sources behind one set of functions:

  * ``postgres`` -- the real source. Every metric is computed as a SQL
    aggregate so the multi-million-row `transactions` table never lands in
    pandas.
  * ``csv``      -- a fallback for exploring the raw dataset without a live
    DB. Loads a capped sample and computes the same frames in pandas.
    Model-side metrics (score, flag, SHAP) are unavailable here, since those
    live only in `model_outputs`.

Column-name resolution
----------------------
The repo is inconsistent about the physical column names: `src/db/schema.py`
maps camelCase (`transID`, `oldbalanceOrg`, `newbalancDest`) while
`src/training/data_extraction.py` and `src/tools/sql_query_tool.py` both
query snake_case (`transaction_id`, `oldbalance_org`, `newbalance_dest`).
Rather than pick one and break on the other, this module reads
`information_schema.columns` once and maps each canonical name onto whatever
the live database actually calls it -- including the `newbalancDest` typo.
Every generated identifier is double-quoted, so either convention works.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import text  # noqa: E402

TRANSACTIONS_TABLE = "transactions"
MODEL_OUTPUTS_TABLE = "model_outputs"

DEFAULT_CSV_PATH = _REPO_ROOT / "notebooks" / "data.csv"

# Canonical name -> every physical spelling seen in this repo. Matching is
# case-insensitive; the typo'd `newbalancDest` from schema.py is included on
# purpose so a DB created from that ORM still resolves.
_TRANSACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "transaction_id": ("transaction_id", "transid", "trans_id", "transactionid"),
    "step": ("step",),
    "type": ("type", "tx_type", "transaction_type"),
    "amount": ("amount",),
    "name_orig": ("name_orig", "nameorig", "orig_name"),
    "oldbalance_org": ("oldbalance_org", "oldbalanceorg", "old_balance_org", "oldbalanceorig"),
    "newbalance_orig": ("newbalance_orig", "newbalanceorig", "new_balance_orig"),
    "name_dest": ("name_dest", "namedest", "dest_name"),
    "oldbalance_dest": ("oldbalance_dest", "oldbalancedest", "old_balance_dest"),
    "newbalance_dest": ("newbalance_dest", "newbalancedest", "newbalancdest", "new_balance_dest"),
    "is_flagged_fraud": ("is_flagged_fraud", "isflaggedfraud", "flagged_fraud"),
    "is_fraud": ("is_fraud", "isfraud", "fraud"),
    "created_at": ("created_at", "createdat"),
}

_MODEL_OUTPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "transaction_id": ("transaction_id", "transid", "trans_id", "transactionid"),
    "score": ("score", "fraud_score", "probability"),
    "flag": ("flag", "is_flagged", "model_flag"),
    "shap_values": ("shap_values", "shapvalues", "shap"),
    "created_at": ("created_at", "createdat"),
}

# CSV header -> canonical. The CSV has no transaction id; one is synthesised.
_CSV_ALIASES = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "oldbalance_org",
    "newbalanceOrig": "newbalance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "oldbalance_dest",
    "newbalanceDest": "newbalance_dest",
    "newbalancDest": "newbalance_dest",
    "isFlaggedFraud": "is_flagged_fraud",
    "isFraud": "is_fraud",
}

# Transaction types whose `amount` debits the originator. CASH_IN credits it,
# so the ledger identity flips sign for that one type.
_DEBIT_TYPES = ("CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

NUMERIC_FEATURES = (
    "amount",
    "oldbalance_org",
    "newbalance_orig",
    "oldbalance_dest",
    "newbalance_dest",
)

STEPS_PER_DAY = 24  # `step` is hours elapsed since the start of the simulation


class SchemaError(RuntimeError):
    """Raised when a required table or column cannot be resolved."""


class ColumnMap:
    """Canonical column name -> quoted physical identifier for one table."""

    def __init__(self, table: str, physical: Iterable[str], aliases: dict[str, tuple[str, ...]]):
        self.table = table
        self._physical = list(physical)
        lowered = {c.lower(): c for c in self._physical}
        self._resolved: dict[str, str] = {}
        for canonical, candidates in aliases.items():
            for cand in candidates:
                if cand.lower() in lowered:
                    self._resolved[canonical] = lowered[cand.lower()]
                    break

    def has(self, canonical: str) -> bool:
        return canonical in self._resolved

    def raw(self, canonical: str) -> str:
        """Physical (unquoted) column name."""
        if canonical not in self._resolved:
            raise SchemaError(
                f"Table {self.table!r} has no column matching {canonical!r}. "
                f"Present columns: {', '.join(self._physical)}"
            )
        return self._resolved[canonical]

    def q(self, canonical: str, prefix: str | None = None) -> str:
        """Double-quoted identifier, optionally table-qualified."""
        col = f'"{self.raw(canonical)}"'
        return f"{prefix}.{col}" if prefix else col

    @property
    def resolved(self) -> dict[str, str]:
        return dict(self._resolved)

    @property
    def missing_required(self) -> list[str]:
        required = ("transaction_id", "step", "type", "amount", "name_orig",
                    "name_dest", "oldbalance_org", "newbalance_orig",
                    "oldbalance_dest", "newbalance_dest", "is_fraud")
        return [k for k in required if k not in self._resolved]


def _truthy(expr: str) -> str:
    """
    SQL that evaluates `expr` as a boolean regardless of its storage type.

    `is_fraud` / `is_flagged_fraud` are boolean in the ORM but 0/1 in the
    source CSV, so a DB loaded straight from the CSV may hold either. Casting
    to text and comparing covers both without a per-column type lookup.
    """
    return f"(COALESCE({expr}::text, '0') IN ('1', 't', 'true', 'TRUE', 'True'))"


def _count_true(expr: str, alias: str) -> str:
    return f"COUNT(*) FILTER (WHERE {_truthy(expr)}) AS {alias}"


class AmlData:
    """
    Uniform metric API over either PostgreSQL or the raw CSV.

    Every public method returns a pandas DataFrame (or a plain dict for
    `kpi_summary`) with identical columns from either source, so the
    dashboard layer never branches on the source itself.
    """

    def __init__(self, source: str = "postgres", csv_path: str | Path | None = None,
                 csv_rows: int = 400_000):
        self.source = source
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        self.csv_rows = csv_rows
        self._engine = None
        self._tx: ColumnMap | None = None
        self._mo: ColumnMap | None = None
        self._frame: pd.DataFrame | None = None
        self._connect()

    # ------------------------------------------------------------------ setup
    def _connect(self) -> None:
        if self.source == "csv":
            self._frame = self._load_csv()
            return

        from src.db.connection import get_engine

        self._engine = get_engine()
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_name IN (:t1, :t2)"
                ),
                {"t1": TRANSACTIONS_TABLE, "t2": MODEL_OUTPUTS_TABLE},
            ).fetchall()

        tx_cols = [r[1] for r in rows if r[0] == TRANSACTIONS_TABLE]
        mo_cols = [r[1] for r in rows if r[0] == MODEL_OUTPUTS_TABLE]
        if not tx_cols:
            raise SchemaError(
                f"Table {TRANSACTIONS_TABLE!r} was not found in the connected database. "
                "Check DATABASE_URL in .env, or switch the dashboard to the CSV source."
            )
        self._tx = ColumnMap(TRANSACTIONS_TABLE, tx_cols, _TRANSACTION_ALIASES)
        self._mo = ColumnMap(MODEL_OUTPUTS_TABLE, mo_cols, _MODEL_OUTPUT_ALIASES) if mo_cols else None
        if self._tx.missing_required:
            raise SchemaError(
                f"Resolved {TRANSACTIONS_TABLE!r} but these canonical columns are missing: "
                f"{self._tx.missing_required}. Present: {', '.join(tx_cols)}"
            )

    def _load_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise SchemaError(f"CSV source not found at {self.csv_path}")
        df = pd.read_csv(self.csv_path, nrows=self.csv_rows)
        df = df.rename(columns={k: v for k, v in _CSV_ALIASES.items() if k in df.columns})
        df["transaction_id"] = [f"CSV{i:09d}" for i in range(len(df))]
        for col in ("is_fraud", "is_flagged_fraud"):
            if col in df.columns:
                df[col] = df[col].astype("float").fillna(0).astype(bool)
        return df

    # ------------------------------------------------------------ capabilities
    @property
    def has_model_outputs(self) -> bool:
        """True when scores/flags are queryable (PostgreSQL source only)."""
        return self.source == "postgres" and self._mo is not None and self._mo.has("score")

    @property
    def has_shap(self) -> bool:
        return self.has_model_outputs and self._mo is not None and self._mo.has("shap_values")

    def describe_source(self) -> str:
        if self.source == "csv":
            return f"CSV sample · {self.csv_path.name} · first {self.csv_rows:,} rows"
        spellings = ", ".join(
            f"{c} → {self._tx.raw(c)}" for c in ("transaction_id", "oldbalance_org", "newbalance_dest")
        )
        return f"PostgreSQL · resolved {spellings}"

    def _read(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    def _csv(self) -> pd.DataFrame:
        if self._frame is None:
            raise SchemaError("CSV frame is not loaded.")
        return self._frame

    def _join_clause(self) -> str:
        """`model_outputs mo JOIN transactions t` on whatever the id column is called."""
        t, m = self._tx, self._mo
        return (
            f"{MODEL_OUTPUTS_TABLE} mo JOIN {TRANSACTIONS_TABLE} t "
            f"ON {t.q('transaction_id', 't')} = {m.q('transaction_id', 'mo')}"
        )

    # =========================================================== 1. portfolio
    def kpi_summary(self) -> dict:
        """
        Headline counters for the KPI row.

        Returns keys: n_tx, total_amount, avg_amount, median_amount,
        n_originators, n_destinations, n_labelled, n_fraud, fraud_base_rate,
        n_rule_flagged, rule_flag_rate, max_step, n_scored, n_model_flagged,
        model_flag_rate, avg_score. Model-side keys are None when
        `model_outputs` is unavailable.
        """
        if self.source == "csv":
            df = self._csv()
            out = dict(
                n_tx=int(len(df)),
                total_amount=float(df["amount"].sum()),
                avg_amount=float(df["amount"].mean()),
                median_amount=float(df["amount"].median()),
                n_originators=int(df["name_orig"].nunique()),
                n_destinations=int(df["name_dest"].nunique()),
                n_labelled=int(df["is_fraud"].notna().sum()),
                n_fraud=int(df["is_fraud"].fillna(False).sum()),
                n_rule_flagged=int(df["is_flagged_fraud"].fillna(False).sum())
                if "is_flagged_fraud" in df.columns else 0,
                max_step=int(df["step"].max()),
            )
        else:
            t = self._tx
            rule_flag = (
                _count_true(t.q("is_flagged_fraud"), "n_rule_flagged")
                if t.has("is_flagged_fraud") else "0::bigint AS n_rule_flagged"
            )
            sql = f"""
                SELECT
                    COUNT(*)                                   AS n_tx,
                    COALESCE(SUM({t.q('amount')}), 0)          AS total_amount,
                    AVG({t.q('amount')})                       AS avg_amount,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {t.q('amount')}) AS median_amount,
                    COUNT(DISTINCT {t.q('name_orig')})         AS n_originators,
                    COUNT(DISTINCT {t.q('name_dest')})         AS n_destinations,
                    COUNT({t.q('is_fraud')})                   AS n_labelled,
                    {_count_true(t.q('is_fraud'), 'n_fraud')},
                    {rule_flag},
                    MAX({t.q('step')})                         AS max_step
                FROM {TRANSACTIONS_TABLE}
            """
            row = self._read(sql).iloc[0].to_dict()
            out = {k: (None if pd.isna(v) else v) for k, v in row.items()}

        out["fraud_base_rate"] = (out["n_fraud"] / out["n_labelled"]) if out.get("n_labelled") else None
        out["rule_flag_rate"] = (out["n_rule_flagged"] / out["n_tx"]) if out.get("n_tx") else None
        out.update(n_scored=None, n_model_flagged=None, model_flag_rate=None, avg_score=None)

        if self.has_model_outputs:
            m = self._mo
            flag_count = (
                _count_true(m.q("flag"), "n_model_flagged")
                if m.has("flag") else "NULL::bigint AS n_model_flagged"
            )
            sql = f"""
                SELECT COUNT(*) AS n_scored,
                       AVG({m.q('score')}) AS avg_score,
                       {flag_count}
                FROM {MODEL_OUTPUTS_TABLE}
            """
            row = self._read(sql).iloc[0]
            out["n_scored"] = int(row["n_scored"])
            out["avg_score"] = None if pd.isna(row["avg_score"]) else float(row["avg_score"])
            out["n_model_flagged"] = None if pd.isna(row["n_model_flagged"]) else int(row["n_model_flagged"])
            if out["n_scored"] and out["n_model_flagged"] is not None:
                out["model_flag_rate"] = out["n_model_flagged"] / out["n_scored"]
        return out

    def volume_by_day(self) -> pd.DataFrame:
        """
        Per-simulation-day volume, value and label counts.

        Columns: day, n_tx, total_amount, n_fraud, n_rule_flagged, fraud_rate.
        """
        if self.source == "csv":
            df = self._csv().assign(day=lambda d: d["step"] // STEPS_PER_DAY)
            out = df.groupby("day", as_index=False).agg(
                n_tx=("amount", "size"),
                total_amount=("amount", "sum"),
                n_fraud=("is_fraud", "sum"),
                n_rule_flagged=("is_flagged_fraud", "sum"),
            )
        else:
            t = self._tx
            rule_flag = (
                _count_true(t.q("is_flagged_fraud"), "n_rule_flagged")
                if t.has("is_flagged_fraud") else "0::bigint AS n_rule_flagged"
            )
            sql = f"""
                SELECT ({t.q('step')} / {STEPS_PER_DAY}) AS day,
                       COUNT(*) AS n_tx,
                       SUM({t.q('amount')}) AS total_amount,
                       {_count_true(t.q('is_fraud'), 'n_fraud')},
                       {rule_flag}
                FROM {TRANSACTIONS_TABLE}
                GROUP BY 1 ORDER BY 1
            """
            out = self._read(sql)
        out["n_fraud"] = out["n_fraud"].astype(float)
        out["n_rule_flagged"] = out["n_rule_flagged"].astype(float)
        out["fraud_rate"] = out["n_fraud"] / out["n_tx"]
        return out

    def hourly_profile(self) -> pd.DataFrame:
        """
        Volume and fraud rate by hour-of-day (`step` mod 24).

        Columns: hour, n_tx, n_fraud, fraud_rate.
        """
        if self.source == "csv":
            df = self._csv().assign(hour=lambda d: d["step"] % STEPS_PER_DAY)
            out = df.groupby("hour", as_index=False).agg(
                n_tx=("amount", "size"), n_fraud=("is_fraud", "sum")
            )
        else:
            t = self._tx
            sql = f"""
                SELECT MOD({t.q('step')}, {STEPS_PER_DAY}) AS hour,
                       COUNT(*) AS n_tx,
                       {_count_true(t.q('is_fraud'), 'n_fraud')}
                FROM {TRANSACTIONS_TABLE}
                GROUP BY 1 ORDER BY 1
            """
            out = self._read(sql)
        out["n_fraud"] = out["n_fraud"].astype(float)
        out["fraud_rate"] = out["n_fraud"] / out["n_tx"]
        return out

    def mix_by_type(self) -> pd.DataFrame:
        """
        Per-channel mix.

        Columns: type, n_tx, total_amount, avg_amount, n_fraud, fraud_rate,
        n_rule_flagged.
        """
        if self.source == "csv":
            df = self._csv()
            out = df.groupby("type", as_index=False).agg(
                n_tx=("amount", "size"),
                total_amount=("amount", "sum"),
                avg_amount=("amount", "mean"),
                n_fraud=("is_fraud", "sum"),
                n_rule_flagged=("is_flagged_fraud", "sum"),
            )
        else:
            t = self._tx
            rule_flag = (
                _count_true(t.q("is_flagged_fraud"), "n_rule_flagged")
                if t.has("is_flagged_fraud") else "0::bigint AS n_rule_flagged"
            )
            sql = f"""
                SELECT {t.q('type')} AS type,
                       COUNT(*) AS n_tx,
                       SUM({t.q('amount')}) AS total_amount,
                       AVG({t.q('amount')}) AS avg_amount,
                       {_count_true(t.q('is_fraud'), 'n_fraud')},
                       {rule_flag}
                FROM {TRANSACTIONS_TABLE}
                GROUP BY 1 ORDER BY 2 DESC
            """
            out = self._read(sql)
        out["n_fraud"] = out["n_fraud"].astype(float)
        out["fraud_rate"] = out["n_fraud"] / out["n_tx"]
        return out.sort_values("n_tx", ascending=False).reset_index(drop=True)

    def balance_summary(self) -> dict:
        """
        Average account balances, before and after a transaction, for both
        the sending account (originator) and the receiving account (dest).

        This dataset has no literal "debt" or "income" column -- these
        before/after balances are the closest equivalent: how much an
        account held going in and coming out of the average transaction.

        Returns:
            {avg_oldbalance_org, avg_newbalance_orig,
             avg_oldbalance_dest, avg_newbalance_dest}
        """
        if self.source == "csv":
            df = self._csv()
            return dict(
                avg_oldbalance_org=float(df["oldbalance_org"].mean()),
                avg_newbalance_orig=float(df["newbalance_orig"].mean()),
                avg_oldbalance_dest=float(df["oldbalance_dest"].mean()),
                avg_newbalance_dest=float(df["newbalance_dest"].mean()),
            )
        t = self._tx
        sql = f"""
            SELECT AVG({t.q('oldbalance_org')})  AS avg_oldbalance_org,
                   AVG({t.q('newbalance_orig')}) AS avg_newbalance_orig,
                   AVG({t.q('oldbalance_dest')}) AS avg_oldbalance_dest,
                   AVG({t.q('newbalance_dest')}) AS avg_newbalance_dest
            FROM {TRANSACTIONS_TABLE}
        """
        row = self._read(sql).iloc[0]
        return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}

    def amount_bands(self) -> pd.DataFrame:
        """
        Volume and fraud rate per order-of-magnitude amount band.

        Powers of ten are the only readable binning for an amount column
        spanning several orders of magnitude. Columns: band, band_order,
        n_tx, total_amount, n_fraud, fraud_rate.
        """
        edges = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
        labels = ["< 100", "100 – 1K", "1K – 10K", "10K – 100K",
                  "100K – 1M", "1M – 10M", "≥ 10M"]
        if self.source == "csv":
            df = self._csv().copy()
            df["band_order"] = np.digitize(df["amount"], edges, right=False)
            out = df.groupby("band_order", as_index=False).agg(
                n_tx=("amount", "size"),
                total_amount=("amount", "sum"),
                n_fraud=("is_fraud", "sum"),
            )
        else:
            t = self._tx
            case = " ".join(f"WHEN {t.q('amount')} < {hi} THEN {i}" for i, hi in enumerate(edges))
            sql = f"""
                SELECT CASE {case} ELSE {len(edges)} END AS band_order,
                       COUNT(*) AS n_tx,
                       SUM({t.q('amount')}) AS total_amount,
                       {_count_true(t.q('is_fraud'), 'n_fraud')}
                FROM {TRANSACTIONS_TABLE}
                GROUP BY 1 ORDER BY 1
            """
            out = self._read(sql)
        out["band_order"] = out["band_order"].astype(int)
        out["band"] = out["band_order"].map(dict(enumerate(labels)))
        out["n_fraud"] = out["n_fraud"].astype(float)
        out["fraud_rate"] = out["n_fraud"] / out["n_tx"]
        return out.sort_values("band_order").reset_index(drop=True)

    # ========================================================== 2. typologies
    def typology_counts(self, large_amount: float = 200_000.0) -> pd.DataFrame:
        """
        Rule-style behavioural patterns counted over the whole table.

        These are descriptive account-behaviour patterns, not verdicts -- they
        are context an analyst reads alongside the model's score, never a
        label applied to a transaction. `hit_rate` is the share of each
        pattern's rows that carry a positive ground-truth label, so a pattern
        with high volume and a near-zero hit rate is noise, not signal.

        Columns: typology, n_tx, n_fraud, hit_rate, share_of_book.
        """
        debit_list = ", ".join(f"'{x}'" for x in _DEBIT_TYPES)
        large_label = f"Amount ≥ {large_amount:,.0f}"

        if self.source == "csv":
            d = self._csv()
            is_debit = d["type"].isin(_DEBIT_TYPES)
            expected_new = np.where(
                is_debit,
                d["oldbalance_org"] - d["amount"],
                d["oldbalance_org"] + d["amount"],
            )
            conds = {
                "Originator balance fully drained":
                    (d["newbalance_orig"] == 0) & (d["oldbalance_org"] > 0)
                    & (d["amount"] >= d["oldbalance_org"]),
                "Destination balances both zero":
                    (d["amount"] > 0) & (d["oldbalance_dest"] == 0) & (d["newbalance_dest"] == 0),
                "Ledger identity broken (originator)":
                    (d["newbalance_orig"] - expected_new).abs() > 0.01,
                "Amount exceeds originator balance":
                    is_debit & (d["amount"] > d["oldbalance_org"]),
                "Round-number amount (multiple of 1K)":
                    (d["amount"] > 0) & (d["amount"] % 1000 == 0),
                large_label:
                    d["amount"] >= large_amount,
            }
            out = pd.DataFrame([
                dict(typology=name, n_tx=int(mask.sum()),
                     n_fraud=float(d.loc[mask, "is_fraud"].sum()))
                for name, mask in conds.items()
            ])
            total = len(d)
        else:
            t = self._tx
            expected_new = (
                f"CASE WHEN {t.q('type')} IN ({debit_list}) "
                f"THEN {t.q('oldbalance_org')} - {t.q('amount')} "
                f"ELSE {t.q('oldbalance_org')} + {t.q('amount')} END"
            )
            conds = {
                "Originator balance fully drained":
                    f"{t.q('newbalance_orig')} = 0 AND {t.q('oldbalance_org')} > 0 "
                    f"AND {t.q('amount')} >= {t.q('oldbalance_org')}",
                "Destination balances both zero":
                    f"{t.q('amount')} > 0 AND {t.q('oldbalance_dest')} = 0 "
                    f"AND {t.q('newbalance_dest')} = 0",
                "Ledger identity broken (originator)":
                    f"ABS({t.q('newbalance_orig')} - ({expected_new})) > 0.01",
                "Amount exceeds originator balance":
                    f"{t.q('type')} IN ({debit_list}) AND {t.q('amount')} > {t.q('oldbalance_org')}",
                "Round-number amount (multiple of 1K)":
                    f"{t.q('amount')} > 0 AND MOD({t.q('amount')}::numeric, 1000) = 0",
                large_label:
                    f"{t.q('amount')} >= {large_amount}",
            }
            selects = ["COUNT(*) AS total_rows"]
            for i, cond in enumerate(conds.values()):
                selects.append(f"COUNT(*) FILTER (WHERE {cond}) AS n_{i}")
                selects.append(
                    f"COUNT(*) FILTER (WHERE ({cond}) AND {_truthy(t.q('is_fraud'))}) AS f_{i}"
                )
            row = self._read(f"SELECT {', '.join(selects)} FROM {TRANSACTIONS_TABLE}").iloc[0]
            total = int(row["total_rows"])
            out = pd.DataFrame([
                dict(typology=name, n_tx=int(row[f"n_{i}"]), n_fraud=float(row[f"f_{i}"]))
                for i, name in enumerate(conds)
            ])

        out["hit_rate"] = np.where(out["n_tx"] > 0, out["n_fraud"] / out["n_tx"], np.nan)
        out["share_of_book"] = out["n_tx"] / total if total else np.nan
        return out.sort_values("n_tx", ascending=False).reset_index(drop=True)

    def dest_fan_in(self, limit: int = 20, min_origins: int = 2) -> pd.DataFrame:
        """
        Destination accounts receiving from many distinct originators -- the
        fan-in shape that mule networks produce.

        Columns: name_dest, n_origins, n_tx, total_amount, n_fraud.
        """
        if self.source == "csv":
            d = self._csv()
            out = d.groupby("name_dest", as_index=False).agg(
                n_origins=("name_orig", "nunique"),
                n_tx=("amount", "size"),
                total_amount=("amount", "sum"),
                n_fraud=("is_fraud", "sum"),
            )
            out = out[out["n_origins"] >= min_origins]
            return (out.sort_values(["n_origins", "total_amount"], ascending=False)
                       .head(limit).reset_index(drop=True))

        t = self._tx
        sql = f"""
            SELECT {t.q('name_dest')} AS name_dest,
                   COUNT(DISTINCT {t.q('name_orig')}) AS n_origins,
                   COUNT(*) AS n_tx,
                   SUM({t.q('amount')}) AS total_amount,
                   {_count_true(t.q('is_fraud'), 'n_fraud')}
            FROM {TRANSACTIONS_TABLE}
            GROUP BY 1
            HAVING COUNT(DISTINCT {t.q('name_orig')}) >= :min_origins
            ORDER BY n_origins DESC, total_amount DESC
            LIMIT :limit
        """
        return self._read(sql, {"min_origins": min_origins, "limit": limit})

    # ======================================================= 3. model outputs
    def score_bins(self, n_bins: int = 200) -> pd.DataFrame:
        """
        Labelled score histogram -- the input to a binned PR curve.

        Binning server-side keeps the PR curve O(n_bins) in pandas instead of
        pulling millions of (score, label) pairs over the wire.

        Columns: bin, score_low, score_high, n_total, n_pos, n_neg.
        """
        empty = pd.DataFrame(columns=["bin", "score_low", "score_high",
                                      "n_total", "n_pos", "n_neg"])
        if not self.has_model_outputs:
            return empty
        t, m = self._tx, self._mo
        sql = f"""
            SELECT WIDTH_BUCKET(LEAST(GREATEST({m.q('score', 'mo')}, 0.0), 1.0 - 1e-9),
                                0.0, 1.0, :n_bins) AS bin,
                   COUNT(*) AS n_total,
                   {_count_true(t.q('is_fraud', 't'), 'n_pos')}
            FROM {self._join_clause()}
            WHERE {t.q('is_fraud', 't')} IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """
        df = self._read(sql, {"n_bins": n_bins})
        if df.empty:
            return empty
        width = 1.0 / n_bins
        df["score_low"] = (df["bin"] - 1) * width
        df["score_high"] = df["bin"] * width
        df["n_pos"] = df["n_pos"].astype(float)
        df["n_neg"] = df["n_total"].astype(float) - df["n_pos"]
        return df

    def score_by_day(self) -> pd.DataFrame:
        """
        Mean/p95 score and model-flag rate per simulation day -- the series a
        score-drift check reads.

        Columns: day, n_scored, avg_score, p95_score, n_model_flagged,
        flag_rate.
        """
        cols = ["day", "n_scored", "avg_score", "p95_score", "n_model_flagged", "flag_rate"]
        if not self.has_model_outputs:
            return pd.DataFrame(columns=cols)
        t, m = self._tx, self._mo
        flag_expr = (
            _count_true(m.q("flag", "mo"), "n_model_flagged")
            if m.has("flag") else "NULL::bigint AS n_model_flagged"
        )
        sql = f"""
            SELECT ({t.q('step', 't')} / {STEPS_PER_DAY}) AS day,
                   COUNT(*) AS n_scored,
                   AVG({m.q('score', 'mo')}) AS avg_score,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {m.q('score', 'mo')}) AS p95_score,
                   {flag_expr}
            FROM {self._join_clause()}
            GROUP BY 1 ORDER BY 1
        """
        df = self._read(sql)
        if df.empty:
            return pd.DataFrame(columns=cols)
        if df["n_model_flagged"].notna().any():
            df["flag_rate"] = df["n_model_flagged"].astype(float) / df["n_scored"]
        else:
            df["flag_rate"] = np.nan
        return df

    def review_queue(self, threshold: float, limit: int = 50) -> pd.DataFrame:
        """
        Highest-scoring transactions at or above `threshold` -- the analyst's
        work queue, ordered newest-and-highest first.

        Columns: transaction_id, score, type, amount, name_orig, name_dest,
        oldbalance_org, newbalance_orig, is_flagged_fraud, is_fraud.
        """
        if not self.has_model_outputs:
            return pd.DataFrame()
        t, m = self._tx, self._mo
        order = f"{m.q('created_at', 'mo')} DESC NULLS LAST, " if m.has("created_at") else ""
        rule_col = (
            f"{t.q('is_flagged_fraud', 't')} AS is_flagged_fraud"
            if t.has("is_flagged_fraud") else "NULL AS is_flagged_fraud"
        )
        sql = f"""
            SELECT {m.q('transaction_id', 'mo')}  AS transaction_id,
                   {m.q('score', 'mo')}           AS score,
                   {t.q('type', 't')}             AS type,
                   {t.q('amount', 't')}           AS amount,
                   {t.q('name_orig', 't')}        AS name_orig,
                   {t.q('name_dest', 't')}        AS name_dest,
                   {t.q('oldbalance_org', 't')}   AS oldbalance_org,
                   {t.q('newbalance_orig', 't')}  AS newbalance_orig,
                   {rule_col},
                   {t.q('is_fraud', 't')}         AS is_fraud
            FROM {self._join_clause()}
            WHERE {m.q('score', 'mo')} >= :threshold
            ORDER BY {order}{m.q('score', 'mo')} DESC
            LIMIT :limit
        """
        return self._read(sql, {"threshold": threshold, "limit": limit})

    def shap_global(self, threshold: float | None = None,
                    sample_rows: int = 20_000) -> pd.DataFrame:
        """
        Global feature attribution: mean |SHAP| and mean signed SHAP per
        feature, over a sample of scored rows.

        mean |SHAP| ranks *how much* each feature moves the score; mean signed
        SHAP shows which way it typically pushes it. Together they are the
        explanation the agent's report is grounded in.

        Columns: feature, mean_abs_shap, mean_shap, n_rows.
        """
        cols = ["feature", "mean_abs_shap", "mean_shap", "n_rows"]
        if not self.has_shap:
            return pd.DataFrame(columns=cols)
        m = self._mo
        where = [f"{m.q('shap_values')} IS NOT NULL"]
        if threshold is not None:
            where.append(f"{m.q('score')} >= :threshold")
        sql = f"""
            WITH sampled AS (
                SELECT {m.q('shap_values')} AS shap_values
                FROM {MODEL_OUTPUTS_TABLE}
                WHERE {' AND '.join(where)}
                LIMIT :sample_rows
            )
            SELECT kv.key AS feature,
                   AVG(ABS(kv.value::numeric)) AS mean_abs_shap,
                   AVG(kv.value::numeric)      AS mean_shap,
                   COUNT(*)                    AS n_rows
            FROM sampled s, JSONB_EACH_TEXT(s.shap_values::jsonb) AS kv
            GROUP BY 1 ORDER BY 2 DESC
        """
        params: dict = {"sample_rows": sample_rows}
        if threshold is not None:
            params["threshold"] = threshold
        try:
            df = self._read(sql, params)
        except Exception as exc:
            # Non-numeric or non-object SHAP payloads: surface rather than raise,
            # so one bad row never takes the whole dashboard down.
            out = pd.DataFrame(columns=cols)
            out.attrs["error"] = str(exc)
            return out
        for col in ("mean_abs_shap", "mean_shap"):
            df[col] = df[col].astype(float)
        return df

    def shap_for_transaction(self, transaction_id: str) -> pd.DataFrame:
        """
        Per-feature SHAP contribution for one transaction, largest magnitude
        first -- the per-case view behind a generated report.

        Columns: feature, shap_value.
        """
        cols = ["feature", "shap_value"]
        if not self.has_shap:
            return pd.DataFrame(columns=cols)
        m = self._mo
        sql = f"""
            SELECT kv.key AS feature, kv.value::numeric AS shap_value
            FROM {MODEL_OUTPUTS_TABLE} mo,
                 JSONB_EACH_TEXT(mo.{'"' + m.raw('shap_values') + '"'}::jsonb) AS kv
            WHERE {m.q('transaction_id', 'mo')} = :tid
            ORDER BY ABS(kv.value::numeric) DESC
        """
        try:
            df = self._read(sql, {"tid": transaction_id})
        except Exception as exc:
            out = pd.DataFrame(columns=cols)
            out.attrs["error"] = str(exc)
            return out
        if not df.empty:
            df["shap_value"] = df["shap_value"].astype(float)
        return df

    # ======================================================== 4. model health
    def feature_histogram(self, feature: str, step_split: int,
                          n_bins: int = 20) -> pd.DataFrame:
        """
        Two aligned histograms of one numeric feature: a reference window
        (`step` < `step_split`) and a current window (`step` >= `step_split`).

        Shared bin edges are what make the two windows comparable -- and what
        lets PSI be computed from the result. This is a model-health signal,
        read separately from any transaction-level flag.

        Columns: bin, bin_low, bin_high, reference_n, current_n.
        """
        if feature not in NUMERIC_FEATURES:
            raise ValueError(
                f"{feature!r} is not one of the numeric features of `transactions`: "
                f"{NUMERIC_FEATURES}"
            )

        if self.source == "csv":
            d = self._csv()
            lo, hi = float(d[feature].min()), float(d[feature].max())
            if hi <= lo:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, n_bins + 1)
            ref_n, _ = np.histogram(d.loc[d["step"] < step_split, feature], bins=edges)
            cur_n, _ = np.histogram(d.loc[d["step"] >= step_split, feature], bins=edges)
            return pd.DataFrame({
                "bin": np.arange(1, n_bins + 1),
                "bin_low": edges[:-1],
                "bin_high": edges[1:],
                "reference_n": ref_n.astype(float),
                "current_n": cur_n.astype(float),
            })

        t = self._tx
        col = t.q(feature)
        bounds = self._read(
            f"SELECT MIN({col}) AS lo, MAX({col}) AS hi FROM {TRANSACTIONS_TABLE}"
        ).iloc[0]
        lo = float(bounds["lo"] or 0.0)
        hi = float(bounds["hi"] or 0.0)
        if hi <= lo:
            hi = lo + 1.0
        sql = f"""
            SELECT WIDTH_BUCKET(LEAST(GREATEST({col}, :lo), :hi - 1e-9), :lo, :hi, :n_bins) AS bin,
                   COUNT(*) FILTER (WHERE {t.q('step')} <  :split) AS reference_n,
                   COUNT(*) FILTER (WHERE {t.q('step')} >= :split) AS current_n
            FROM {TRANSACTIONS_TABLE}
            GROUP BY 1 ORDER BY 1
        """
        df = self._read(sql, {"lo": lo, "hi": hi, "n_bins": n_bins, "split": step_split})
        width = (hi - lo) / n_bins
        full = pd.DataFrame({"bin": np.arange(1, n_bins + 1)})
        df = full.merge(df, on="bin", how="left").fillna({"reference_n": 0, "current_n": 0})
        df["bin_low"] = lo + (df["bin"] - 1) * width
        df["bin_high"] = lo + df["bin"] * width
        df["reference_n"] = df["reference_n"].astype(float)
        df["current_n"] = df["current_n"].astype(float)
        return df[["bin", "bin_low", "bin_high", "reference_n", "current_n"]]
