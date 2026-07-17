"""Data profiling: summary statistics, missing values, dtype analysis."""

from __future__ import annotations

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "dtypes": {},
        "missing": {},
        "numeric_stats": {},
    }
    for col in df.columns:
        profile["dtypes"][col] = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        if missing > 0:
            profile["missing"][col] = {"count": missing, "pct": round(missing / len(df) * 100, 2)}
        if df[col].dtype.kind in ("i", "f"):
            profile["numeric_stats"][col] = {
                "min": float(df[col].min()) if df[col].notna().any() else None,
                "max": float(df[col].max()) if df[col].notna().any() else None,
                "mean": float(df[col].mean()) if df[col].notna().any() else None,
                "std": float(df[col].std()) if df[col].notna().any() else None,
            }
    return profile


def detect_issues(df: pd.DataFrame) -> list[dict]:
    issues = []
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            issues.append({"column": col, "type": "missing_values", "count": missing})
        if df[col].dtype.kind in ("i", "f"):
            z = (df[col] - df[col].mean()).abs() / df[col].std()
            if z.gt(3).sum() > 0:
                issues.append({"column": col, "type": "outliers_zscore", "count": int(z.gt(3).sum())})
    return issues
