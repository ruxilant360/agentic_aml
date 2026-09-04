"""
Macro-averaged Precision-Recall evaluation for the XGBoost fraud classifier.

Macro-PR (not accuracy, not plain ROC-AUC) is the project's evaluation
metric of record for this imbalanced binary problem: it averages the PR-AUC
computed independently for each class (fraud / not-fraud), so performance
on the minority (fraud) class isn't swamped by the majority class.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def compute_macro_pr(y_true, y_scores) -> dict:
    """
    Compute macro-averaged precision-recall metrics for a binary classifier.

    Args:
        y_true: Ground-truth binary labels (0/1), array-like.
        y_scores: Predicted fraud scores/probabilities, array-like.

    Returns:
        {
          "pr_auc_positive": float,  # PR-AUC for the fraud (1) class
          "pr_auc_negative": float,  # PR-AUC for the not-fraud (0) class
          "macro_pr_auc": float,     # average of the two
          "precision": np.ndarray,   # precision curve for the positive class
          "recall": np.ndarray,      # recall curve for the positive class
          "thresholds": np.ndarray,
        }
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    pr_auc_positive = average_precision_score(y_true, y_scores)
    # PR-AUC for the negative class: flip labels and scores.
    pr_auc_negative = average_precision_score(1 - y_true, 1 - y_scores)

    macro_pr_auc = (pr_auc_positive + pr_auc_negative) / 2.0

    return {
        "pr_auc_positive": pr_auc_positive,
        "pr_auc_negative": pr_auc_negative,
        "macro_pr_auc": macro_pr_auc,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }


# ---- tests ----

def test_macro_pr_perfect_separation():
    y_true = [0, 0, 0, 1, 1, 1]
    y_scores = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
    result = compute_macro_pr(y_true, y_scores)
    assert result["pr_auc_positive"] == 1.0
    assert result["pr_auc_negative"] == 1.0
    assert result["macro_pr_auc"] == 1.0


def test_macro_pr_is_average_of_both_classes():
    y_true = [0, 0, 0, 0, 1, 1]
    y_scores = [0.1, 0.9, 0.2, 0.3, 0.6, 0.7]
    result = compute_macro_pr(y_true, y_scores)
    expected = (result["pr_auc_positive"] + result["pr_auc_negative"]) / 2.0
    assert result["macro_pr_auc"] == expected


def test_macro_pr_worse_than_random_is_reflected_low():
    y_true = [1, 1, 1, 0, 0, 0]
    # Scores are inversely correlated with the label.
    y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    result = compute_macro_pr(y_true, y_scores)
    assert result["pr_auc_positive"] < 0.6
