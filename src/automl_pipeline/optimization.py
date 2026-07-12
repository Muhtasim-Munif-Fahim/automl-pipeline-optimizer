"""Hyperparameter optimization using grid search and random search."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 10, 20, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
    },
    "ridge": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "elastic_net": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "kernel": ["rbf", "poly", "sigmoid"],
    },
}


def get_param_grid(model_name: str, n_iter: int = 20) -> dict[str, list[Any]] | None:
    return PARAM_GRIDS.get(model_name)


def optimize_grid(model, param_grid: dict, X_train, y_train, cv: int = 5, scoring: str = "accuracy"):
    search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring,
        n_jobs=-1, verbose=0, return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def optimize_random(model, param_dist: dict, X_train, y_train, n_iter: int = 20, cv: int = 5, scoring: str = "accuracy"):
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=cv,
        scoring=scoring, n_jobs=-1, verbose=0,
        random_state=42, return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def get_best_params(search_result) -> dict[str, Any]:
    return search_result.best_params_


def get_search_results(search_result) -> dict[str, Any]:
    results = search_result.cv_results_
    return {
        "best_params": search_result.best_params_,
        "best_score": float(search_result.best_score_),
        "all_scores": [float(s) for s in results["mean_test_score"]],
        "all_params": results["params"],
    }
