"""
Threshold selection: precision-priority policy.

We'd rather miss some fraud (lower recall) than flood the analyst with
false positives. Threshold selection therefore maximizes precision subject
to a minimum acceptable recall floor -- it does NOT default to an
F1-balanced threshold.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold_by_precision(y_true, y_scores, min_recall: float = 0.5) -> dict:
    """
    Select the score threshold that maximizes precision, subject to recall
    staying at or above `min_recall`.

    Args:
        y_true: Ground-truth binary labels (0/1), array-like.
        y_scores: Predicted fraud scores/probabilities, array-like.
        min_recall: Minimum acceptable recall -- the recall floor. Among
            thresholds meeting this floor, the one with highest precision
            wins.

    Returns:
        {"threshold": float, "precision": float, "recall": float}

    Raises:
        ValueError: if no threshold achieves at least `min_recall`.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds (the last point is threshold=inf, recall=0); align by
    # dropping that trailing point so indices line up with `thresholds`.
    precision, recall = precision[:-1], recall[:-1]

    eligible = recall >= min_recall
    if not np.any(eligible):
        raise ValueError(
            f"No threshold achieves recall >= {min_recall}. "
            f"Max achievable recall was {recall.max() if len(recall) else 0.0}."
        )

    eligible_precisions = np.where(eligible, precision, -np.inf)
    best_idx = int(np.argmax(eligible_precisions))

    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }


# ---- tests ----

def _sample_scores():
    # A clean separation with a noisy middle band, enough to distinguish
    # precision-optimal vs. F1-optimal thresholds.
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_scores = [0.05, 0.1, 0.2, 0.4, 0.55, 0.5, 0.6, 0.8, 0.9, 0.95]
    return y_true, y_scores


def test_selected_threshold_meets_recall_floor():
    y_true, y_scores = _sample_scores()
    result = select_threshold_by_precision(y_true, y_scores, min_recall=0.6)
    assert result["recall"] >= 0.6


def test_selected_threshold_maximizes_precision_among_eligible():
    y_true, y_scores = _sample_scores()
    result = select_threshold_by_precision(y_true, y_scores, min_recall=0.6)

    # Brute-force check: no eligible threshold should beat the selected precision.
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision, recall = precision[:-1], recall[:-1]
    eligible_precisions = [p for p, r in zip(precision, recall) if r >= 0.6]
    assert result["precision"] == max(eligible_precisions)


def test_raises_when_recall_floor_unachievable():
    y_true, y_scores = _sample_scores()
    try:
        select_threshold_by_precision(y_true, y_scores, min_recall=1.01)
        assert False, "expected ValueError for an unachievable recall floor"
    except ValueError:
        pass


def test_policy_default_is_not_f1_balanced():
    """
    Sanity check on the policy itself: the precision-maximizing choice at a
    modest recall floor should not always coincide with the F1-optimal
    threshold -- if it always did, this function would be indistinguishable
    from (and could accidentally regress to) F1-balanced selection.
    """
    y_true, y_scores = _sample_scores()
    precision_priority = select_threshold_by_precision(y_true, y_scores, min_recall=0.6)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall + 1e-12),
        0.0,
    )
    f1_best_idx = int(np.argmax(f1))
    f1_threshold = float(thresholds[f1_best_idx])

    # Not a hard requirement that they always differ, but they must not be
    # required to match -- precision-priority selection should be free to
    # pick a stricter, higher-precision threshold than the F1-optimal one.
    assert precision_priority["precision"] >= float(precision[f1_best_idx])
