"""
AML monitoring dashboard (Streamlit).

Six views over the two PostgreSQL tables this project writes to:

  Portfolio         volume, value and channel mix of the book
  Review queue      what the model is sending to an analyst, and how much
  Model performance macro-PR, and the precision-priority threshold choice
  SHAP drivers      which features move the score, and in which direction
  Behaviour         descriptive account-behaviour patterns, with hit rates
  Model health      covariate shift (PSI) and score drift over time

Run it from the repo root:

    streamlit run dashboards/streamlit_app.py

Framing note: this dashboard reports model scores, flags and SHAP
attributions. It does not label a transaction as anything -- a flag is a
routing signal and the SHAP view is the evidence behind it; the conclusion is
the analyst's to draw.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboards import metrics  # noqa: E402
from dashboards.data_access import NUMERIC_FEATURES, AmlData, SchemaError  # noqa: E402
from dashboards.theme import Theme  # noqa: E402

st.set_page_config(page_title="AML monitoring", page_icon="\N{BAR CHART}", layout="wide")

FEATURE_LABELS = {
    "amount": "Amount",
    "oldbalance_org": "Originator balance (before)",
    "newbalance_orig": "Originator balance (after)",
    "oldbalance_dest": "Destination balance (before)",
    "newbalance_dest": "Destination balance (after)",
}

PSI_ICONS = {"good": "\N{HEAVY CHECK MARK}", "warning": "\N{WARNING SIGN}", "critical": "\N{HEAVY EXCLAMATION MARK SYMBOL}"}
PSI_WORDS = {"good": "stable", "warning": "moderate shift", "critical": "significant shift"}


# --------------------------------------------------------------- formatting
def fmt_int(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{int(v):,}"


def fmt_money(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    v = float(v)
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= cutoff:
            return f"{v / cutoff:,.2f}{suffix}"
    return f"{v:,.0f}"


def fmt_pct(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v) * 100:.{digits}f}%"


def _width_kwargs() -> dict:
    """
    Pick the full-width argument this Streamlit understands.

    1.49 introduced `width="stretch"` and deprecated `use_container_width`.
    A try/except is not enough to tell them apart: on older versions
    `st.plotly_chart` swallows unknown keywords into `**kwargs` instead of
    raising, and `st.dataframe` takes `width` as an integer -- so a wrong
    guess fails silently rather than loudly. Compare the version instead.
    """
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
    except (AttributeError, ValueError):
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}


_WIDTH = _width_kwargs()


def show(fig: go.Figure) -> None:
    """Render a Plotly figure full-width."""
    st.plotly_chart(fig, config={"displaylogo": False}, **_WIDTH)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    """Render a dataframe full-width."""
    st.dataframe(df, **_WIDTH, **kwargs)


def table_relief(df: pd.DataFrame, label: str = "Show underlying rows") -> None:
    """
    The relief rule: three light-mode categorical hues sit below 3:1 against
    the light surface, so any chart using them ships a table view alongside.
    """
    with st.expander(label):
        show_table(df)


# ------------------------------------------------------------------- caching
@st.cache_resource(show_spinner="Connecting to the AML database…")
def get_data(source: str, csv_rows: int) -> AmlData:
    return AmlData(source=source, csv_rows=csv_rows)


# In both helpers below, `_d` is underscore-prefixed so Streamlit skips
# hashing it (an AmlData holds a live DB engine and is not hashable), and
# `key` carries the source identity instead. `key` is never read in the body
# on purpose -- it exists so switching source invalidates the cache. Don't
# remove it as "unused".
@st.cache_data(ttl=600, show_spinner="Aggregating…")
def q_kpis(_d: AmlData, key: str) -> dict:
    return _d.kpi_summary()


@st.cache_data(ttl=600, show_spinner="Aggregating…")
def q_frame(_d: AmlData, key: str, method: str, **kwargs) -> pd.DataFrame:
    """One cached entry point for every frame-returning method on AmlData."""
    return getattr(_d, method)(**kwargs)


# --------------------------------------------------------------- chart forms
def line_chart(df: pd.DataFrame, x: str, series: dict, theme: Theme, title: str,
               x_title: str = "", y_title: str = "", y_tickformat: str | None = None,
               hover_fmt: str = ":,.0f") -> go.Figure:
    """
    One or more 2px lines on a single axis.

    `series` maps column name -> legend label. A legend appears as soon as
    there are two series; a single series needs none, because the title
    already names it. Never two y-scales: callers with mismatched units build
    two charts instead.
    """
    fig = go.Figure()
    for i, (col, label) in enumerate(series.items()):
        colour = theme.categorical[i % len(theme.categorical)]
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], name=label, mode="lines",
            line=dict(color=colour, width=2, shape="linear"),
            hovertemplate=f"{label}<br>%{{x}} · %{{y{hover_fmt}}}<extra></extra>",
        ))
    layout = theme.layout(title=dict(text=title), showlegend=len(series) > 1,
                          hovermode="x unified" if len(series) > 1 else "closest")
    # Set only the label text -- replacing the whole `title` dict would drop
    # the font and standoff the theme put there.
    layout["xaxis"]["title"]["text"] = x_title
    layout["yaxis"]["title"]["text"] = y_title
    if y_tickformat:
        layout["yaxis"]["tickformat"] = y_tickformat
    fig.update_layout(**layout)
    return fig


def hbar_chart(df: pd.DataFrame, category: str, value: str, theme: Theme, title: str,
               value_labels: list[str] | None = None, colors: list[str] | None = None,
               x_title: str = "", x_tickformat: str | None = None) -> go.Figure:
    """
    Horizontal bars with rounded data-ends and selective direct labels.

    Horizontal because the categories here (transaction types, feature names,
    behaviour patterns) have long names that would be unreadable rotated. One
    hue for one series unless `colors` supplies per-entity identity hues --
    never a value-ramp over nominal categories. Bars are separated by the
    layout's bar gap, not by a border drawn around each mark.
    """
    # Plotly draws the first category at the bottom of a horizontal axis, so
    # reverse the frame to keep the caller's ordering reading top-down.
    d = df.iloc[::-1]
    labels = list(reversed(value_labels)) if value_labels is not None \
        else [f"{v:,.0f}" for v in d[value]]
    fig = go.Figure(go.Bar(
        x=d[value], y=d[category], orientation="h",
        marker=dict(color=list(reversed(colors)) if colors else theme.accent,
                    cornerradius=4),
        text=labels, textposition="outside", cliponaxis=False,
        textfont=dict(color=theme.chrome["text_secondary"], size=12),
        hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
    ))
    layout = theme.layout(
        title=dict(text=title), showlegend=False,
        # Extra right margin: `automargin` reserves room for tick labels and
        # axis titles, but not for bar text sitting outside the bar end.
        margin=dict(r=110),
        height=max(260, 36 * len(d) + 150),
    )
    layout["xaxis"].update(showgrid=True, gridcolor=theme.chrome["gridline"], showline=False)
    layout["xaxis"]["title"]["text"] = x_title
    if x_tickformat:
        layout["xaxis"]["tickformat"] = x_tickformat
    layout["yaxis"].update(showgrid=False)
    fig.update_layout(**layout)
    return fig


def diverging_hbar(df: pd.DataFrame, category: str, value: str, theme: Theme,
                   title: str, x_title: str = "") -> go.Figure:
    """
    Diverging horizontal bars: warm/cool poles around a neutral zero line.

    The right form when the data's job is *polarity* -- here, whether a
    feature pushed the score up or down. Blue = pushed down, red = pushed up;
    the midpoint is neutral gray, never a hue.
    """
    d = df.iloc[::-1]
    cool, _, warm = theme.diverging()
    fig = go.Figure(go.Bar(
        x=d[value], y=d[category], orientation="h",
        marker=dict(color=[warm if v >= 0 else cool for v in d[value]], cornerradius=4),
        text=[f"{v:+.4f}" for v in d[value]], textposition="outside", cliponaxis=False,
        textfont=dict(color=theme.chrome["text_secondary"], size=12),
        hovertemplate="%{y}<br>SHAP %{x:+.4f}<extra></extra>",
    ))
    layout = theme.layout(title=dict(text=title), showlegend=False,
                          margin=dict(r=110),
                          height=max(280, 36 * len(d) + 150))
    layout["xaxis"].update(showgrid=True, gridcolor=theme.chrome["gridline"], showline=False,
                           zeroline=True, zerolinecolor=theme.chrome["baseline"], zerolinewidth=1)
    layout["xaxis"]["title"]["text"] = x_title
    # Symmetric range, padded for the outside value labels. Symmetry keeps
    # equal magnitudes equally long on both sides of zero -- the whole point
    # of a diverging scale -- and the padding stops a negative bar's label
    # from running into the category labels on the left.
    magnitude = float(np.nanmax(np.abs(d[value]))) if len(d) else 1.0
    if not np.isfinite(magnitude) or magnitude == 0:
        magnitude = 1.0
    layout["xaxis"]["range"] = [-magnitude * 1.45, magnitude * 1.45]
    layout["yaxis"].update(showgrid=False)
    fig.update_layout(**layout)
    return fig


# ------------------------------------------------------------------ 1. book
def tab_portfolio(d: AmlData, theme: Theme, key: str) -> None:
    kpis = q_kpis(d, key)

    st.markdown(
        f"<div style='font-size:52px;line-height:1.05;font-weight:600;"
        f"color:{theme.chrome['text_primary']}'>{fmt_int(kpis['n_tx'])}</div>"
        f"<div style='color:{theme.chrome['text_secondary']};margin-bottom:18px'>"
        f"transactions in the book · {fmt_money(kpis['total_amount'])} total value</div>",
        unsafe_allow_html=True,
    )

    c = st.columns(5)
    c[0].metric("Labelled rows", fmt_int(kpis["n_labelled"]))
    c[1].metric("Positive-label rate", fmt_pct(kpis["fraud_base_rate"], 3),
                help="Ground-truth base rate. The imbalance that makes macro-PR the metric of record.")
    c[2].metric("Rule-flag rate", fmt_pct(kpis["rule_flag_rate"], 3),
                help="Share carrying the pre-existing `is_flagged_fraud` rule flag.")
    c[3].metric("Distinct originators", fmt_int(kpis["n_originators"]))
    c[4].metric("Median amount", fmt_money(kpis["median_amount"]))

    st.divider()

    daily = q_frame(d, key, "volume_by_day")
    left, right = st.columns(2)
    with left:
        # Count and value have different units, so they get their own axis in
        # their own chart -- never two y-scales on one plot.
        show(line_chart(daily, "day", {"n_tx": "Transactions"}, theme,
                        "Transactions per day", x_title="Simulation day",
                        y_title="count"))
    with right:
        show(line_chart(daily, "day", {"total_amount": "Value"}, theme,
                        "Value moved per day", x_title="Simulation day",
                        y_title="amount", hover_fmt=":,.0f"))

    hourly = q_frame(d, key, "hourly_profile")
    left, right = st.columns(2)
    with left:
        show(line_chart(hourly, "hour", {"n_tx": "Transactions"}, theme,
                        "Volume by hour of day", x_title="Hour (step mod 24)",
                        y_title="count"))
    with right:
        show(line_chart(hourly, "hour", {"fraud_rate": "Positive-label rate"}, theme,
                        "Positive-label rate by hour of day",
                        x_title="Hour (step mod 24)", y_title="rate",
                        y_tickformat=".2%", hover_fmt=":.3%"))

    st.divider()

    mix = q_frame(d, key, "mix_by_type")
    type_colors = theme.type_colors()
    left, right = st.columns(2)
    with left:
        # `type` is an identity, so each channel keeps its own hue everywhere.
        show(hbar_chart(mix, "type", "n_tx", theme, "Volume by channel",
                        value_labels=[fmt_int(v) for v in mix["n_tx"]],
                        colors=[type_colors.get(t, theme.accent) for t in mix["type"]],
                        x_title="transactions"))
    with right:
        show(hbar_chart(mix, "type", "fraud_rate", theme,
                        "Positive-label rate by channel",
                        value_labels=[fmt_pct(v, 3) for v in mix["fraud_rate"]],
                        colors=[type_colors.get(t, theme.accent) for t in mix["type"]],
                        x_title="rate", x_tickformat=".2%"))
    table_relief(mix, "Channel mix — underlying rows")

    bands = q_frame(d, key, "amount_bands")
    left, right = st.columns(2)
    with left:
        show(hbar_chart(bands, "band", "n_tx", theme, "Volume by amount band",
                        value_labels=[fmt_int(v) for v in bands["n_tx"]],
                        x_title="transactions"))
    with right:
        show(hbar_chart(bands, "band", "fraud_rate", theme,
                        "Positive-label rate by amount band",
                        value_labels=[fmt_pct(v, 3) for v in bands["fraud_rate"]],
                        x_title="rate", x_tickformat=".2%"))
    table_relief(bands, "Amount bands — underlying rows")


# ---------------------------------------------------------- 2. review queue
def tab_queue(d: AmlData, theme: Theme, key: str, threshold: float) -> None:
    if not d.has_model_outputs:
        st.info(
            "No `model_outputs` rows are reachable, so there is no review queue to show. "
            "Run the inference pipeline (or switch the source to PostgreSQL in the sidebar)."
        )
        return

    kpis = q_kpis(d, key)
    bins = q_frame(d, key, "score_bins", n_bins=200)
    pr = metrics.pr_curve_from_bins(bins)

    c = st.columns(4)
    c[0].metric("Scored transactions", fmt_int(kpis["n_scored"]))
    c[1].metric("Carrying a model flag", fmt_int(kpis["n_model_flagged"]),
                help="`flag` column on model_outputs, as written by the inference pipeline.")
    c[2].metric("Flag rate", fmt_pct(kpis["model_flag_rate"], 3))
    c[3].metric("Mean score", "—" if kpis["avg_score"] is None else f"{kpis['avg_score']:.4f}")

    st.divider()

    if not pr.empty:
        left, right = st.columns(2)
        with left:
            show(line_chart(pr.sort_values("threshold"), "threshold",
                            {"n_alerts": "Cases at or above cut"}, theme,
                            "Queue size by score cut", x_title="score threshold",
                            y_title="cases"))
        with right:
            show(line_chart(pr.sort_values("threshold"), "threshold",
                            {"precision": "Precision"}, theme,
                            "Queue precision by score cut", x_title="score threshold",
                            y_title="precision", y_tickformat=".0%", hover_fmt=":.2%"))
        st.caption(
            "Read these together: moving the cut right shrinks the queue and raises its "
            "precision. The precision-priority policy picks the rightmost cut that still "
            "clears the recall floor."
        )

    daily = q_frame(d, key, "score_by_day")
    if not daily.empty and daily["flag_rate"].notna().any():
        show(line_chart(daily, "day", {"flag_rate": "Flag rate"}, theme,
                        "Flag rate per day", x_title="Simulation day", y_title="rate",
                        y_tickformat=".2%", hover_fmt=":.3%"))

    st.divider()
    st.subheader(f"Top of queue at score ≥ {threshold:.3f}")
    queue = q_frame(d, key, "review_queue", threshold=threshold, limit=50)
    if queue.empty:
        st.info(f"No scored transaction reaches {threshold:.3f}. Lower the cut in the sidebar.")
    else:
        show_table(queue, hide_index=True)
        st.caption(
            "A row here is a routing decision, not a verdict — open the SHAP drivers tab "
            "for the per-feature evidence behind any one of these scores."
        )


# ------------------------------------------------------- 3. model performance
def tab_model(d: AmlData, theme: Theme, key: str, min_recall: float, threshold: float) -> None:
    if not d.has_model_outputs:
        st.info("Model performance needs `model_outputs` joined to labelled transactions.")
        return

    bins = q_frame(d, key, "score_bins", n_bins=200)
    if bins.empty:
        st.warning(
            "`model_outputs` has rows, but none join to a transaction with a non-null "
            "`is_fraud` label — performance metrics need ground truth."
        )
        return

    pr = metrics.pr_curve_from_bins(bins)
    macro = metrics.macro_pr_auc(bins)
    chosen = metrics.select_threshold_by_precision(pr, min_recall=min_recall)
    f1_choice = metrics.best_f1_threshold(pr)

    c = st.columns(4)
    c[0].metric("Macro-PR AUC", f"{macro['macro_pr_auc']:.4f}",
                help="Mean of the PR-AUC for each class. The project's metric of record.")
    c[1].metric("PR-AUC (positive class)", f"{macro['pr_auc_positive']:.4f}")
    c[2].metric("PR-AUC (negative class)", f"{macro['pr_auc_negative']:.4f}")
    c[3].metric("Base rate", fmt_pct(macro["base_rate"], 3))

    st.divider()
    st.subheader("Precision-priority threshold")
    st.caption(
        f"Policy: maximise precision subject to recall ≥ {min_recall:.0%}. Recall is a "
        "constraint, not the objective — a missed case costs less than a queue full of "
        "false positives. This is deliberately not the F1-balanced cut."
    )

    if chosen is None:
        st.error(
            f"No score cut reaches recall ≥ {min_recall:.0%}. "
            f"Max achievable recall is {pr['recall'].max():.1%} — lower the floor in the sidebar."
        )
    else:
        c = st.columns(5)
        c[0].metric("Threshold", f"{chosen['threshold']:.4f}")
        c[1].metric("Precision", fmt_pct(chosen["precision"]))
        c[2].metric("Recall", fmt_pct(chosen["recall"]))
        c[3].metric("Cases queued", fmt_int(chosen["n_alerts"]))
        c[4].metric("Share of book queued", fmt_pct(chosen["alert_rate"], 3))

        if f1_choice is not None:
            delta_p = chosen["precision"] - f1_choice["precision"]
            st.caption(
                f"For contrast, the F1-balanced cut would sit at {f1_choice['threshold']:.4f} "
                f"with precision {fmt_pct(f1_choice['precision'])} and recall "
                f"{fmt_pct(f1_choice['recall'])} — {abs(delta_p) * 100:.1f} points "
                f"{'less' if delta_p > 0 else 'more'} precision and "
                f"{fmt_int(f1_choice['n_alerts'])} cases queued. Shown only as the cost of "
                "the policy; it is not the recommendation."
            )

    st.divider()
    left, right = st.columns(2)
    with left:
        fig = line_chart(pr.sort_values("recall"), "recall", {"precision": "Precision"},
                         theme, "Precision-recall curve (positive class)",
                         x_title="recall", y_title="precision", y_tickformat=".0%",
                         hover_fmt=":.2%")
        fig.update_xaxes(tickformat=".0%")
        if not np.isnan(macro["base_rate"]):
            fig.add_hline(y=macro["base_rate"], line_color=theme.chrome["muted"],
                          line_width=1, annotation_text="base rate (random)",
                          annotation_position="top left",
                          annotation_font=dict(color=theme.chrome["muted"], size=11))
        if chosen is not None:
            fig.add_trace(go.Scatter(
                x=[chosen["recall"]], y=[chosen["precision"]], mode="markers+text",
                marker=dict(size=10, color=theme.status["good"],
                            line=dict(color=theme.chrome["surface"], width=2)),
                text=["chosen cut"], textposition="bottom right",
                textfont=dict(color=theme.chrome["text_secondary"], size=11),
                name="chosen cut", showlegend=False,
                hovertemplate=f"chosen cut {chosen['threshold']:.4f}<br>"
                              f"precision %{{y:.2%}} · recall %{{x:.2%}}<extra></extra>",
            ))
        show(fig)

    with right:
        # 200 bins is the resolution the PR curve needs, but it is far too
        # fine for a distribution: the minority class has few enough rows
        # that each bin is either empty or a spike. Fold 4:1 for this chart.
        dist = (bins.assign(_group=(bins["bin"] - 1) // 4)
                    .groupby("_group", as_index=False)
                    .agg(score_low=("score_low", "min"),
                         n_pos=("n_pos", "sum"),
                         n_neg=("n_neg", "sum")))
        total_pos = dist["n_pos"].sum()
        total_neg = dist["n_neg"].sum()
        dist["share_pos"] = dist["n_pos"] / total_pos if total_pos else np.nan
        dist["share_neg"] = dist["n_neg"] / total_neg if total_neg else np.nan
        # Shares, not counts: the two classes differ by orders of magnitude, and
        # normalising within each class puts both on one honest axis.
        show(line_chart(dist.sort_values("score_low"), "score_low",
                        {"share_neg": "Label 0 (no fraud)", "share_pos": "Label 1 (fraud)"},
                        theme, "Score distribution within each class",
                        x_title="score", y_title="share of class",
                        y_tickformat=".1%", hover_fmt=":.3%"))

    cut = threshold if chosen is None else chosen["threshold"]
    cm = metrics.confusion_at(pr, cut, total_negatives=float(bins["n_neg"].sum()))
    if cm:
            st.subheader(f"Confusion matrix at score ≥ {cm['threshold']:.4f}")
            show_table(pd.DataFrame(
                [[fmt_int(cm["tn"]), fmt_int(cm["fp"])],
                 [fmt_int(cm["fn"]), fmt_int(cm["tp"])]],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            ))
            st.caption(
                f"Precision {fmt_pct(cm['precision'])} · recall {fmt_pct(cm['recall'])} · "
                f"false-positive rate {fmt_pct(cm['false_positive_rate'], 4)}."
            )

    st.divider()
    lift = metrics.lift_table(bins)
    if not lift.empty:
        left, right = st.columns([3, 2])
        with left:
            show(hbar_chart(lift.assign(label=lambda x: "Decile " + x["decile"].astype(str)),
                            "label", "lift", theme,
                            "Lift over base rate by score decile",
                            value_labels=[f"{v:,.1f}×" for v in lift["lift"]],
                            x_title="lift"))
        with right:
            show_table(
                lift.assign(
                    hit_rate=lambda x: x["hit_rate"].map(lambda v: fmt_pct(v, 3)),
                    cumulative_hit_rate=lambda x: x["cumulative_hit_rate"].map(lambda v: fmt_pct(v, 3)),
                    lift=lambda x: x["lift"].map(lambda v: f"{v:,.1f}×"),
                    n_tx=lambda x: x["n_tx"].map(fmt_int),
                    n_pos=lambda x: x["n_pos"].map(fmt_int),
                ),
                hide_index=True,
            )
        st.caption(
            "Capacity view: if the team can work N cases a day, the cumulative hit rate "
            "is what they would see working the top N scores."
        )


# -------------------------------------------------------------- 4. shap view
def tab_shap(d: AmlData, theme: Theme, key: str, threshold: float) -> None:
    if not d.has_shap:
        st.info(
            "No `shap_values` payloads are reachable. The inference pipeline writes them "
            "per scored transaction; once it has run, this tab fills in."
        )
        return

    scope = st.radio("Scope", ["Flagged cases only", "All scored cases"],
                     horizontal=True, label_visibility="collapsed")
    cut = threshold if scope.startswith("Flagged") else None

    glob = q_frame(d, key, "shap_global", threshold=cut, sample_rows=20_000)
    if glob.empty:
        err = glob.attrs.get("error")
        st.warning("No SHAP rows matched." + (f" Query error: {err}" if err else ""))
        return

    top = glob.head(14)
    left, right = st.columns(2)
    with left:
        show(hbar_chart(top.assign(label=lambda x: x["feature"].map(
                            lambda f: FEATURE_LABELS.get(f, f))),
                        "label", "mean_abs_shap", theme,
                        "How much each feature moves the score",
                        value_labels=[f"{v:.4f}" for v in top["mean_abs_shap"]],
                        x_title="mean |SHAP|"))
        st.caption("Magnitude only — this ranks influence, not direction.")
    with right:
        show(diverging_hbar(top.assign(label=lambda x: x["feature"].map(
                                lambda f: FEATURE_LABELS.get(f, f))),
                            "label", "mean_shap", theme,
                            "Which way each feature pushes the score",
                            x_title="mean signed SHAP"))
        st.caption("Red pushes the score up, blue pushes it down; the zero line is neutral.")

    table_relief(glob, "Feature attributions — underlying rows")

    st.divider()
    st.subheader("Single case")
    queue = q_frame(d, key, "review_queue", threshold=threshold, limit=50)
    if queue.empty:
        st.info("No scored case reaches the current cut, so there is nothing to break down.")
        return

    options = queue["transaction_id"].tolist()
    chosen_id = st.selectbox(
        "Transaction", options,
        format_func=lambda t: f"{t} · score "
                              f"{queue.loc[queue['transaction_id'] == t, 'score'].iloc[0]:.4f}",
    )
    detail = q_frame(d, key, "shap_for_transaction", transaction_id=chosen_id)
    if detail.empty:
        st.info("No SHAP payload stored for this transaction.")
        return

    row = queue.loc[queue["transaction_id"] == chosen_id].iloc[0]
    c = st.columns(4)
    c[0].metric("Score", f"{row['score']:.4f}")
    c[1].metric("Channel", str(row["type"]))
    c[2].metric("Amount", fmt_money(row["amount"]))
    c[3].metric("Originator balance before", fmt_money(row["oldbalance_org"]))

    show(diverging_hbar(
        detail.head(14).assign(label=lambda x: x["feature"].map(lambda f: FEATURE_LABELS.get(f, f))),
        "label", "shap_value", theme,
        f"Feature contributions for {chosen_id}", x_title="SHAP contribution",
    ))
    st.caption(
        "These contributions are the evidence behind the score. They describe what the "
        "model reacted to; the conclusion about this transaction remains the analyst's."
    )


# --------------------------------------------------------- 5. behaviour view
def tab_behaviour(d: AmlData, theme: Theme, key: str, large_amount: float) -> None:
    typ = q_frame(d, key, "typology_counts", large_amount=large_amount)

    left, right = st.columns(2)
    with left:
        show(hbar_chart(typ, "typology", "share_of_book", theme,
                        "How much of the book each pattern covers",
                        value_labels=[fmt_pct(v, 2) for v in typ["share_of_book"]],
                        x_title="share of all transactions", x_tickformat=".0%"))
    with right:
        show(hbar_chart(typ.sort_values("hit_rate", ascending=False), "typology", "hit_rate",
                        theme, "Positive-label rate within each pattern",
                        value_labels=[fmt_pct(v, 2) for v in
                                      typ.sort_values("hit_rate", ascending=False)["hit_rate"]],
                        x_title="rate", x_tickformat=".2%"))
    st.caption(
        "A pattern with wide coverage and a near-zero label rate is noise, not signal. "
        "These are descriptive account-behaviour patterns an analyst reads next to the "
        "score — none of them is a verdict on a transaction."
    )
    table_relief(typ, "Behaviour patterns — underlying rows")

    st.divider()
    st.subheader("Destination fan-in")
    st.caption(
        "Accounts receiving from many distinct originators. Fan-in is the structural "
        "shape a mule network leaves behind; a high count here is a reason to look, "
        "nothing more."
    )
    fan = q_frame(d, key, "dest_fan_in", limit=20, min_origins=2)
    if fan.empty:
        st.info("No destination account received from more than one originator in this data.")
    else:
        show_table(fan, hide_index=True)


# -------------------------------------------------------- 6. model health view
def tab_health(d: AmlData, theme: Theme, key: str, split_frac: float) -> None:
    kpis = q_kpis(d, key)
    max_step = int(kpis["max_step"] or 0)
    split = max(1, int(max_step * split_frac))

    st.caption(
        f"Reference window: step < {split:,}. Current window: step ≥ {split:,} "
        f"(of {max_step:,} steps). This is a model-health reading, separate from any "
        "transaction-level flag — it says the inputs moved, not that a transaction is wrong."
    )

    rows, hists = [], {}
    for feature in NUMERIC_FEATURES:
        h = q_frame(d, key, "feature_histogram", feature=feature, step_split=split, n_bins=20)
        hists[feature] = h
        value = metrics.psi(h["reference_n"], h["current_n"])
        band = metrics.psi_band(value)
        rows.append(dict(feature=FEATURE_LABELS.get(feature, feature),
                         psi=value, band=band,
                         reading=f"{PSI_ICONS[band]} {PSI_WORDS[band]}"))
    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    # Status colors carry meaning here (stable / moderate / significant), and
    # they ship with an icon plus a word so the hue never carries it alone.
    show(hbar_chart(psi_df, "feature", "psi", theme,
                    "Covariate shift by feature (PSI)",
                    value_labels=[f"{r.psi:.3f}  {r.reading}" for r in psi_df.itertuples()],
                    colors=[theme.status[b] for b in psi_df["band"]],
                    x_title="PSI"))
    st.caption(
        "PSI < 0.1 stable · 0.10–0.25 moderate shift · > 0.25 significant shift. "
        "Complementary to the KS test in `src/inference/covariate_shift.py`: KS gives a "
        "p-value, PSI gives a magnitude that stays meaningful at this sample size."
    )
    table_relief(psi_df[["feature", "psi", "reading"]], "PSI readings — underlying rows")

    st.divider()
    pick = st.selectbox("Inspect one feature's distribution", list(NUMERIC_FEATURES),
                        format_func=lambda f: FEATURE_LABELS.get(f, f))
    h = hists[pick].copy()
    ref_total = h["reference_n"].sum() or 1
    cur_total = h["current_n"].sum() or 1
    h["reference_share"] = h["reference_n"] / ref_total
    h["current_share"] = h["current_n"] / cur_total
    show(line_chart(h, "bin_low",
                    {"reference_share": "Reference window", "current_share": "Current window"},
                    theme, f"{FEATURE_LABELS.get(pick, pick)} — distribution shift",
                    x_title=FEATURE_LABELS.get(pick, pick), y_title="share of window",
                    y_tickformat=".1%", hover_fmt=":.2%"))

    if d.has_model_outputs:
        daily = q_frame(d, key, "score_by_day")
        if not daily.empty:
            st.divider()
            show(line_chart(daily, "day",
                            {"avg_score": "Mean score", "p95_score": "95th percentile score"},
                            theme, "Score drift over time", x_title="Simulation day",
                            y_title="score", hover_fmt=":.4f"))
            st.caption(
                "Both series are scores on the same 0–1 scale, so they share one axis. "
                "A rising p95 with a flat mean means the tail is moving, not the book."
            )


# -------------------------------------------------------------------- shell
def main() -> None:
    st.sidebar.title("AML monitoring")

    source = st.sidebar.radio(
        "Data source", ["PostgreSQL", "CSV sample"],
        help="PostgreSQL is the real source and the only one carrying model outputs. "
             "The CSV sample explores the raw dataset without a live DB.",
    )
    source_key = "postgres" if source == "PostgreSQL" else "csv"

    csv_rows = 400_000
    if source_key == "csv":
        csv_rows = st.sidebar.select_slider(
            "Rows to load", [50_000, 100_000, 200_000, 400_000, 800_000], value=200_000,
            help="The CSV holds ~5M rows; a capped head keeps the app responsive.",
        )

    mode = st.sidebar.radio("Chart palette", ["Light", "Dark"], horizontal=True,
                            help="Each mode has its own validated steps — not an automatic flip. "
                                 "Set base=\"dark\" in .streamlit/config.toml to match the app chrome.")
    theme = Theme(mode.lower())

    threshold = st.sidebar.slider(
        "Queue score cut", 0.0, 1.0, 0.50, 0.01,
        help="The score at or above which a case enters the review queue.",
    )
    min_recall = st.sidebar.slider(
        "Recall floor", 0.05, 0.95, 0.50, 0.05,
        help="Threshold selection maximises precision subject to this floor.",
    )
    large_amount = st.sidebar.number_input(
        "Large-amount pattern threshold", min_value=0.0, value=200_000.0, step=50_000.0,
    )

    try:
        d = get_data(source_key, csv_rows)
    except SchemaError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(
            f"Could not open the data source: {exc}\n\n"
            "Check `DATABASE_URL` in `.env`, or switch to the CSV sample in the sidebar."
        )
        st.stop()

    key = f"{source_key}:{csv_rows}"
    st.sidebar.caption(d.describe_source())
    if not d.has_model_outputs:
        st.sidebar.warning("No model outputs reachable — score, flag and SHAP tabs are inert.")

    tabs = st.tabs(["Portfolio", "Review queue", "Model performance",
                    "SHAP drivers", "Behaviour", "Model health"])
    with tabs[0]:
        tab_portfolio(d, theme, key)
    with tabs[1]:
        tab_queue(d, theme, key, threshold)
    with tabs[2]:
        tab_model(d, theme, key, min_recall, threshold)
    with tabs[3]:
        tab_shap(d, theme, key, threshold)
    with tabs[4]:
        tab_behaviour(d, theme, key, large_amount)
    with tabs[5]:
        tab_health(d, theme, key, split_frac=0.7)


if __name__ == "__main__":
    main()
