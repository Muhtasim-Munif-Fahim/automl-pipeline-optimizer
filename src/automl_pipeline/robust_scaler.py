"""Robust scaler with configurable quantile range."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RobustScalerDF(BaseEstimator, TransformerMixin):
    def __init__(self, quantile_range: tuple[float, float] = (25.0, 75.0)):
        self.quantile_range = quantile_range

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = X.median()
        q_low, q_high = np.percentile(X, self.quantile_range, axis=0)
        self.iqr_ = pd.Series(q_high - q_low, index=X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.medians_) / self.iqr_.replace(0, 1)
