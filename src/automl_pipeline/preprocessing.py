"""Data preprocessing: missing values, encoding, scaling, outlier handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


def impute_missing(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if df[col].dtype.kind in ("i", "f"):
            col_strategy = "median" if strategy == "auto" else strategy
            df[col] = df[col].fillna(
                df[col].median() if col_strategy == "median" else df[col].mean()
            )
        else:
            df[col] = df[col].fillna(
                df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
            )
    return df


def encode_categorical(df: pd.DataFrame, method: str = "auto") -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique() <= 10 or method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def scale_numeric(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) == 0:
        return df
    scaler: Any = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }.get(method, StandardScaler())
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def handle_outliers(
    df: pd.DataFrame, method: str = "clip", threshold: float = 3.0
) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if method == "clip":
            mean, std = df[col].mean(), df[col].std()
            lower, upper = mean - threshold * std, mean + threshold * std
            df[col] = df[col].clip(lower, upper)
        elif method == "remove":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
    return df


def preprocess(
    df: pd.DataFrame, target: str, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    cfg = config or {}
    df = impute_missing(df, cfg.get("impute_strategy", "auto"))
    df = handle_outliers(
        df, cfg.get("outlier_method", "clip"), cfg.get("outlier_threshold", 3.0)
    )
    y = df[target]
    X = df.drop(columns=[target])
    X = encode_categorical(X, cfg.get("encoding_method", "auto"))
    X = scale_numeric(X, cfg.get("scaling_method", "standard"))
    return X, y
