"""Feature selection: variance threshold, mutual information, importance-based."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
from typing import Any


def variance_threshold_selection(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    selector = VarianceThreshold(threshold=threshold)
    selected = selector.fit_transform(X)
    kept = [c for c, v in zip(X.columns, selector.get_support()) if v]
    return X[kept]


def mutual_information_selection(X: pd.DataFrame, y: pd.Series, k: int = 20, problem_type: str = "regression") -> pd.DataFrame:
    mi_func = mutual_info_classif if "classification" in problem_type else mutual_info_regression
    mi = mi_func(X, y, random_state=42)
    top_indices = mi.argsort()[-k:][::-1] if len(mi) > k else mi.argsort()[::-1]
    top_features = [X.columns[i] for i in top_indices if i < len(X.columns)]
    return X[top_features]


def get_feature_importance(model: Any, feature_names: list[str], top_n: int = 20) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        scores = dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, "coef_"):
        coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        scores = dict(zip(feature_names, np.abs(coefs)))
    else:
        return {}
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])
    return sorted_scores
