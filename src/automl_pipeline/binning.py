"""Automated binning for continuous features with multiple strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_width_bins(series: pd.Series, n_bins: int = 10) -> pd.Series:
    return pd.cut(series, bins=n_bins, labels=False)


def quantile_bins(series: pd.Series, n_bins: int = 10) -> pd.Series:
    return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")


def auto_bin(series: pd.Series, strategy: str = "quantile", n_bins: int = 10) -> pd.Series:
    if strategy == "equal_width":
        return equal_width_bins(series, n_bins)
    return quantile_bins(series, n_bins)
