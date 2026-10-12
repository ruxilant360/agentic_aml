"""
Model training via RandomizedSearchCV. All hyperparameter-search
configuration (the search space, iteration count, CV folds, scoring) is
accepted as function arguments -- none of it is hardcoded, so callers can
sweep/tune freely without editing this module.
"""
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.training.model import define_model



def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_distributions: dict,
    base_params: dict | None = None,
    n_iter: int = 15,
    # Imbalanced fraud data: average_precision (area under the PR curve) is
    # the closest built-in scorer to the project's macro-PR evaluation
    # policy -- plain accuracy/ROC-AUC scorers are intentionally not used
    # here. Final threshold selection still happens separately in
    # eval/model per the precision-priority policy.
    scoring: str = "average_precision",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """
    Run a randomized hyperparameter search and fit an XGBoost fraud classifier.

    Args:
        X_train: Training features (post categorical encoding).
        y_train: Training labels (`is_fraud`).
        param_distributions: Hyperparameter search space, e.g.
            {"max_depth": [3, 5, 7], "n_estimators": [100, 300, 500],
             "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.6, 0.8, 1.0],
             "min_child_weight": [1, 3, 5]}.
        base_params: Fixed params passed to `define_model` for the base
            estimator (e.g. {"eval_metric": "aucpr"}), separate from the
            values being searched over.
        n_iter: Number of parameter settings sampled by RandomizedSearchCV.
        cv: Number of cross-validation folds.
        scoring: sklearn scoring string used to rank candidates during search.
        random_state: Seed for the search's sampling and the base estimator.
        n_jobs: Parallelism for the search (-1 = use all cores).

    Returns:
        The fitted RandomizedSearchCV object (`.best_estimator_` holds the
        trained XGBoost model, `.best_params_` the winning hyperparameters).
    """

    cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = random_state)

    base_params = dict(base_params or {})
    base_params.setdefault("random_state", random_state)
    estimator = define_model(base_params)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True
    )
    search.fit(X_train, y_train)
    return search
