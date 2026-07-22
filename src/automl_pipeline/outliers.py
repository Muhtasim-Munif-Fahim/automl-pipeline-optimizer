"""Outlier detection using Isolation Forest and IQR methods."""

from __future__ import annotations

import numpy as np
import pandas as pd


def iqr_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return series.clip(lower, upper)


def isolation_forest_outliers(df: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(df.select_dtypes(include=["int64", "float64"]))
    return pd.Series(preds == -1, name="outlier")
