"""Target encoding for high-cardinality categorical variables."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing: float = 10.0, min_samples_leaf: int = 5):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.maps_: dict[str, pd.Series] = {}
        self.global_mean_ = float(y.mean()) if y is not None else 0.0
        for col in X.select_dtypes(include=["object", "category"]).columns:
            if y is not None:
                stats = pd.DataFrame(
                    {"count": X[col].value_counts(), "mean": y.groupby(X[col]).mean()}
                )
                stats["smooth"] = (
                    stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
                ) / (stats["count"] + self.smoothing)
                self.maps_[col] = stats["smooth"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.maps_.items():
            X[col] = X[col].map(mapping).fillna(self.global_mean_)
        return X
