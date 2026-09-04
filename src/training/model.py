"""
XGBoost model definition. Hyperparameters are always passed in by the
caller -- nothing here bakes in fixed hyperparameter values.
"""

from xgboost import XGBClassifier


def define_model(params: dict | None = None) -> XGBClassifier:
    """
    Construct an (unfit) XGBoost classifier.

    Args:
        params: Hyperparameters to pass through to `XGBClassifier`
            (e.g. `max_depth`, `n_estimators`, `learning_rate`, `subsample`,
            `min_child_weight`). None/omitted keys fall back to XGBoost's
            own defaults -- no hyperparameter values are hardcoded here.

    Returns:
        An unfit XGBClassifier ready to be passed as the estimator into
        RandomizedSearchCV (see `src.training.train.train_model`).
    """
    params = dict(params or {})
    # `eval_metric` and `use_label_encoder` concern how the estimator
    # reports/consumes labels, not tunable model capacity -- fine as
    # overridable defaults rather than search-space hyperparameters.
    params.setdefault("eval_metric", "aucpr")
    params.setdefault("random_state", 42)
    return XGBClassifier(**params)
