"""
Simple transaction dashboard (Streamlit).

One page. Total transfers, average amounts, average account balances,
and a breakdown by transaction type. No model scores, no SHAP, no
statistics jargon -- just the basic numbers.

Run it from the repo root:

    streamlit run dashboards/simple_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboards.data_access import AmlData, SchemaError  # noqa: E402

st.set_page_config(page_title="Transaction Dashboard", page_icon="\N{BAR CHART}", layout="wide")

# Plain colors, one per transaction type -- no fancy theme system, just
# something readable and consistent across the two charts.
TYPE_COLORS = {
    "CASH_IN": "#2a78d6",
    "CASH_OUT": "#eb6834",
    "DEBIT": "#1baf7a",
    "PAYMENT": "#eda100",
    "TRANSFER": "#e87ba4",
}


def money(v) -> str:
    """Format a number as money, shortened for big values (1.2M, 450K, etc.)."""
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= cutoff:
            return f"${v / cutoff:,.2f}{suffix}"
    return f"${v:,.0f}"


def count(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{int(v):,}"


@st.cache_resource(show_spinner="Connecting…")
def get_data(source: str, csv_rows: int) -> AmlData:
    return AmlData(source=source, csv_rows=csv_rows)


@st.cache_data(ttl=600, show_spinner="Loading data…")
def load_kpis(_d: AmlData, key: str) -> dict:
    return _d.kpi_summary()


@st.cache_data(ttl=600, show_spinner="Loading data…")
def load_balances(_d: AmlData, key: str) -> dict:
    return _d.balance_summary()


@st.cache_data(ttl=600, show_spinner="Loading data…")
def load_mix(_d: AmlData, key: str) -> pd.DataFrame:
    return _d.mix_by_type()


def main() -> None:
    st.title("Transaction Dashboard")

    source = st.sidebar.radio("Data source", ["PostgreSQL", "CSV file"])
    source_key = "postgres" if source == "PostgreSQL" else "csv"
    csv_rows = 400_000
    if source_key == "csv":
        csv_rows = st.sidebar.select_slider(
            "Rows to load", [50_000, 100_000, 200_000, 400_000, 800_000], value=200_000
        )

    try:
        d = get_data(source_key, csv_rows)
    except SchemaError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not connect: {exc}\n\nTry switching to the CSV file in the sidebar.")
        st.stop()

    key = f"{source_key}:{csv_rows}"
    kpis = load_kpis(d, key)
    balances = load_balances(d, key)
    mix = load_mix(d, key)

    # --- Top numbers ---------------------------------------------------
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", count(kpis["n_tx"]))
    c2.metric("Total Amount Moved", money(kpis["total_amount"]))
    c3.metric("Average Transaction", money(kpis["avg_amount"]))
    c4.metric("Median Transaction", money(kpis["median_amount"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distinct Senders", count(kpis["n_originators"]))
    c2.metric("Distinct Recipients", count(kpis["n_destinations"]))
    c3.metric("Labelled Fraud Cases", count(kpis["n_fraud"]))
    c4.metric(
        "Fraud Rate",
        "—" if kpis["fraud_base_rate"] is None else f"{kpis['fraud_base_rate'] * 100:.2f}%",
    )

    st.divider()

    # --- Balances (closest thing this data has to "debt / income") -----
    st.subheader("Average Account Balances")
    st.caption("This dataset has no debt or income column — these are the closest equivalent: "
               "how much money was in an account right before and right after a transaction.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sender Balance (Before)", money(balances["avg_oldbalance_org"]))
    c2.metric("Sender Balance (After)", money(balances["avg_newbalance_orig"]))
    c3.metric("Recipient Balance (Before)", money(balances["avg_oldbalance_dest"]))
    c4.metric("Recipient Balance (After)", money(balances["avg_newbalance_dest"]))

    st.divider()

    # --- Breakdown by transaction type ----------------------------------
    st.subheader("By Transaction Type")
    colors = [TYPE_COLORS.get(t, "#999999") for t in mix["type"]]

    left, right = st.columns(2)
    with left:
        fig = px.bar(mix, x="type", y="n_tx", title="Number of Transactions",
                     color="type", color_discrete_sequence=colors)
        fig.update_layout(showlegend=False, yaxis_title="transactions", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(mix, x="type", y="total_amount", title="Total Amount Moved",
                     color="type", color_discrete_sequence=colors)
        fig.update_layout(showlegend=False, yaxis_title="amount", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Type Breakdown — Table")
    table = mix[["type", "n_tx", "total_amount", "avg_amount"]].copy()
    table.columns = ["Type", "Transactions", "Total Amount", "Average Amount"]
    table["Total Amount"] = table["Total Amount"].map(money)
    table["Average Amount"] = table["Average Amount"].map(money)
    table["Transactions"] = table["Transactions"].map(count)
    st.dataframe(table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
