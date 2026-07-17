"""Model definitions and training utilities."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

CLASSIFICATION_MODELS: dict[str, Any] = {
    "logistic_regression": (
        LogisticRegression,
        {"max_iter": 1000, "random_state": 42, "n_jobs": -1},
    ),
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier,
        {"n_estimators": 100, "random_state": 42},
    ),
    "svm": (SVC, {"probability": True, "random_state": 42}),
}

REGRESSION_MODELS: dict[str, Any] = {
    "linear_regression": (LinearRegression, {"n_jobs": -1}),
    "ridge": (Ridge, {"random_state": 42}),
    "lasso": (Lasso, {"random_state": 42}),
    "random_forest": (
        RandomForestRegressor,
        {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    ),
    "gradient_boosting": (
        GradientBoostingRegressor,
        {"n_estimators": 100, "random_state": 42},
    ),
}

CLASSIFICATION_METRICS: dict[str, Any] = {
    "accuracy": accuracy_score,
    "f1": lambda y, p: f1_score(y, p, average="weighted"),
    "precision": lambda y, p: precision_score(
        y, p, average="weighted", zero_division=0
    ),
    "recall": lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
}

REGRESSION_METRICS: dict[str, Any] = {
    "r2": r2_score,
    "rmse": lambda y, p: mean_squared_error(y, p) ** 0.5,
    "mae": mean_absolute_error,
    "explained_variance": explained_variance_score,
}


def get_models(problem_type: str) -> dict[str, Any]:
    if problem_type in ("binary_classification", "multiclass_classification"):
        return dict(CLASSIFICATION_MODELS)
    return dict(REGRESSION_MODELS)


def get_metrics(problem_type: str) -> dict[str, Any]:
    if problem_type in ("binary_classification", "multiclass_classification"):
        return dict(CLASSIFICATION_METRICS)
    return dict(REGRESSION_METRICS)


def train_model(model_name: str, X_train, y_train, **kwargs) -> Any:
    if model_name in CLASSIFICATION_MODELS:
        model_cls, defaults = CLASSIFICATION_MODELS[model_name]
    elif model_name in REGRESSION_MODELS:
        model_cls, defaults = REGRESSION_MODELS[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    params = {**defaults, **kwargs}
    model = model_cls(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, problem_type: str) -> dict[str, float]:
    preds = model.predict(X_test)
    metrics = get_metrics(problem_type)
    results: dict[str, float] = {}
    for name, metric_fn in metrics.items():
        try:
            results[name] = float(metric_fn(y_test, preds))
        except Exception:
            results[name] = 0.0
    if hasattr(model, "predict_proba") and problem_type == "binary_classification":
        try:
            proba = model.predict_proba(X_test)[:, 1]
            results["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass
    return results
