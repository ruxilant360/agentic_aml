"""
Metric computations for the AML dashboards.

Everything here is pure: frames in, frames out, no I/O. The curve functions
take the *binned* score histogram produced by
`data_access.AmlData.score_bins` rather than raw score arrays, so the whole
model-performance tab costs one SQL aggregate regardless of table size.

Policy note
-----------
Threshold selection follows the project policy: **maximise precision subject
to a recall floor**, never "best F1". `select_threshold_by_precision` is the
binned counterpart of the reference implementation in
`eval/model/test_threshold_selection.py`; the ranking rule is identical, only
the resolution differs (one candidate per score bin instead of one per
distinct score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- PR curves
def pr_curve_from_bins(score_bins: pd.DataFrame) -> pd.DataFrame:
    """
    Precision/recall at every bin edge, derived from a binned score histogram.

    Walks the bins from the highest score down, accumulating the positives
    and negatives that a threshold at each bin's lower edge would admit. The
    result is the PR curve for the positive (fraud) class.

    Args:
        score_bins: frame with columns score_low, n_pos, n_neg (as returned
            by `AmlData.score_bins`).

    Returns:
        Frame with columns threshold, tp, fp, fn, precision, recall, f1,
        n_alerts, alert_rate -- one row per bin edge, ordered by descending
        threshold. Empty frame if there is nothing to score.
    """
    cols = ["threshold", "tp", "fp", "fn", "precision", "recall", "f1",
            "n_alerts", "alert_rate"]
    if score_bins is None or score_bins.empty:
        return pd.DataFrame(columns=cols)

    b = score_bins.sort_values("score_low", ascending=False).reset_index(drop=True)
    total_pos = float(b["n_pos"].sum())
    total = float(b["n_pos"].sum() + b["n_neg"].sum())
    if total_pos == 0 or total == 0:
        return pd.DataFrame(columns=cols)

    tp = b["n_pos"].cumsum().astype(float)
    fp = b["n_neg"].cumsum().astype(float)
    n_alerts = tp + fp

    out = pd.DataFrame({
        "threshold": b["score_low"].astype(float),
        "tp": tp,
        "fp": fp,
        "fn": total_pos - tp,
        "n_alerts": n_alerts,
    })
    out["precision"] = np.where(n_alerts > 0, tp / n_alerts, np.nan)
    out["recall"] = tp / total_pos
    denom = out["precision"] + out["recall"]
    out["f1"] = np.where(denom > 0, 2 * out["precision"] * out["recall"] / denom, 0.0)
    out["alert_rate"] = n_alerts / total
    return out[cols]


def select_threshold_by_precision(pr: pd.DataFrame, min_recall: float = 0.5) -> dict | None:
    """
    Pick the threshold that maximises precision subject to recall staying at
    or above `min_recall`.

    This is the project's precision-priority policy: a missed case costs less
    than an analyst queue full of false positives, so recall is a *constraint*
    and precision is the objective. Deliberately not F1-balanced.

    Args:
        pr: PR curve frame from `pr_curve_from_bins`.
        min_recall: the recall floor. Candidates below it are ineligible.

    Returns:
        dict with threshold, precision, recall, f1, tp, fp, fn, n_alerts,
        alert_rate -- or None when no threshold reaches the floor (the caller
        shows the max achievable recall instead of guessing).
    """
    if pr is None or pr.empty:
        return None
    eligible = pr[pr["recall"] >= min_recall]
    if eligible.empty:
        return None
    best = eligible.loc[eligible["precision"].idxmax()]
    return {k: float(best[k]) for k in pr.columns}


def best_f1_threshold(pr: pd.DataFrame) -> dict | None:
    """
    The F1-maximising threshold -- computed only as a *contrast* to the
    precision-priority choice, so the dashboard can show what the policy is
    deliberately giving up. Never the dashboard's recommendation.
    """
    if pr is None or pr.empty:
        return None
    best = pr.loc[pr["f1"].idxmax()]
    return {k: float(best[k]) for k in pr.columns}


def macro_pr_auc(score_bins: pd.DataFrame) -> dict:
    """
    Macro-averaged PR-AUC from the binned histogram: the mean of the PR-AUC
    for the positive (fraud) class and for the negative class.

    Macro-PR, not accuracy or plain ROC-AUC, is this project's metric of
    record -- it keeps minority-class performance from being swamped by the
    majority class.

    Returns:
        {"pr_auc_positive", "pr_auc_negative", "macro_pr_auc", "base_rate"};
        values are NaN when one class is absent.
    """
    out = {"pr_auc_positive": float("nan"), "pr_auc_negative": float("nan"),
           "macro_pr_auc": float("nan"), "base_rate": float("nan")}
    if score_bins is None or score_bins.empty:
        return out

    pos = pr_curve_from_bins(score_bins)
    # Negative class: flip both the label and the score direction, which for a
    # binned histogram means reversing the bins and swapping n_pos / n_neg.
    flipped = score_bins.copy()
    flipped["n_pos"], flipped["n_neg"] = score_bins["n_neg"], score_bins["n_pos"]
    flipped["score_low"] = 1.0 - score_bins["score_high"]
    flipped["score_high"] = 1.0 - score_bins["score_low"]
    neg = pr_curve_from_bins(flipped)

    out["pr_auc_positive"] = _step_auc(pos)
    out["pr_auc_negative"] = _step_auc(neg)
    out["macro_pr_auc"] = float(np.nanmean([out["pr_auc_positive"], out["pr_auc_negative"]]))
    total = float(score_bins["n_pos"].sum() + score_bins["n_neg"].sum())
    out["base_rate"] = float(score_bins["n_pos"].sum()) / total if total else float("nan")
    return out


def _step_auc(pr: pd.DataFrame) -> float:
    """
    Average precision: the step-wise sum of precision weighted by the recall
    gained at each threshold -- the same estimator `average_precision_score`
    uses, so it does not reward a jagged curve the way trapezoidal area does.
    """
    if pr is None or pr.empty:
        return float("nan")
    curve = pr.sort_values("recall").reset_index(drop=True)
    recall_gain = curve["recall"].diff().fillna(curve["recall"].iloc[0])
    return float((curve["precision"] * recall_gain).sum())


def confusion_at(pr: pd.DataFrame, threshold: float, total_negatives: float) -> dict:
    """
    Confusion-matrix counts at the curve point nearest `threshold`.

    Args:
        pr: PR curve frame.
        threshold: the score cut to report at.
        total_negatives: count of negative-label rows, needed for true
            negatives (the PR curve itself never sees them).

    Returns:
        {"threshold", "tp", "fp", "fn", "tn", "precision", "recall",
         "false_positive_rate"}.
    """
    if pr is None or pr.empty:
        return {}
    row = pr.iloc[(pr["threshold"] - threshold).abs().argmin()]
    tn = max(total_negatives - float(row["fp"]), 0.0)
    return {
        "threshold": float(row["threshold"]),
        "tp": float(row["tp"]),
        "fp": float(row["fp"]),
        "fn": float(row["fn"]),
        "tn": tn,
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "false_positive_rate": (float(row["fp"]) / total_negatives) if total_negatives else float("nan"),
    }


# -------------------------------------------------------------- model health
def psi(reference_n: np.ndarray | pd.Series, current_n: np.ndarray | pd.Series,
        epsilon: float = 1e-6) -> float:
    """
    Population Stability Index between two aligned histograms.

    PSI is the dashboard's at-a-glance drift reading, complementary to the
    KS test in `src/inference/covariate_shift.py`: KS gives a p-value, PSI
    gives a magnitude that survives large sample sizes (where KS flags
    everything). Conventional reading: < 0.1 stable, 0.1-0.25 moderate
    shift, > 0.25 significant shift.

    Args:
        reference_n: per-bin counts in the reference (training) window.
        current_n: per-bin counts in the current window, same bin edges.
        epsilon: floor applied to zero-share bins so the log stays finite.

    Returns:
        The PSI value, or NaN if either window is empty.
    """
    ref = np.asarray(reference_n, dtype=float)
    cur = np.asarray(current_n, dtype=float)
    if ref.sum() <= 0 or cur.sum() <= 0:
        return float("nan")
    ref_share = np.clip(ref / ref.sum(), epsilon, None)
    cur_share = np.clip(cur / cur.sum(), epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_band(value: float) -> str:
    """Map a PSI value onto a status band: 'good', 'warning' or 'critical'."""
    if value is None or np.isnan(value):
        return "warning"
    if value < 0.1:
        return "good"
    if value < 0.25:
        return "warning"
    return "critical"


def lift_table(score_bins: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """
    Decile lift over the base rate: how much denser the positive class is in
    each score decile than in the book overall.

    This is the capacity-planning view -- it answers "if the team can work N
    cases a day, what hit rate do the top N scores give us?"

    Columns: decile, n_tx, n_pos, hit_rate, lift, cumulative_hit_rate.
    """
    cols = ["decile", "n_tx", "n_pos", "hit_rate", "lift", "cumulative_hit_rate"]
    if score_bins is None or score_bins.empty:
        return pd.DataFrame(columns=cols)

    b = score_bins.sort_values("score_low", ascending=False).reset_index(drop=True)
    total = float(b["n_pos"].sum() + b["n_neg"].sum())
    base_rate = float(b["n_pos"].sum()) / total if total else float("nan")
    if total == 0:
        return pd.DataFrame(columns=cols)

    b["n_tx"] = b["n_pos"] + b["n_neg"]
    cum = b["n_tx"].cumsum()
    # Assign each bin to the decile its cumulative position falls in, so
    # deciles hold ~equal row counts even with a very skewed score histogram.
    b["decile"] = np.minimum((cum / total * n_deciles).apply(np.ceil).astype(int), n_deciles)
    b.loc[b["decile"] < 1, "decile"] = 1

    out = b.groupby("decile", as_index=False).agg(
        n_tx=("n_tx", "sum"), n_pos=("n_pos", "sum")
    ).sort_values("decile").reset_index(drop=True)
    out["hit_rate"] = out["n_pos"] / out["n_tx"]
    out["lift"] = out["hit_rate"] / base_rate if base_rate else np.nan
    out["cumulative_hit_rate"] = out["n_pos"].cumsum() / out["n_tx"].cumsum()
    return out[cols]
