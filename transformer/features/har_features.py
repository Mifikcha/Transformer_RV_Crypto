"""
HAR and subsampled RV feature engineering for Transformer pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _subsampled_rv(returns: pd.Series, window: int, n_subsamples: int = 4) -> pd.Series:
    estimates: list[pd.Series] = []
    for k in range(n_subsamples):
        shifted = returns.shift(k)
        sq = shifted * shifted
        estimates.append(sq.rolling(window, min_periods=window).sum())
    return pd.concat(estimates, axis=1).mean(axis=1)


def add_har_and_subsampled_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds HAR-style weekly/monthly RV aggregates and subsampled RV features.
    Returns a modified copy.
    """
    out = df.copy()
    har_base_cols = ["realized_vol_15min", "realized_vol_60min", "realized_vol_240min"]

    weekly_window = 5 * 288
    monthly_window = 22 * 288
    for col in har_base_cols:
        if col not in out.columns:
            continue
        out[f"{col}_weekly"] = out[col].rolling(weekly_window, min_periods=weekly_window).mean()
        out[f"{col}_monthly"] = out[col].rolling(monthly_window, min_periods=monthly_window).mean()

    if "log_return_5min" in out.columns:
        out["rv_subsampled_15min"] = _subsampled_rv(out["log_return_5min"], window=3, n_subsamples=4)
        out["rv_subsampled_60min"] = _subsampled_rv(out["log_return_5min"], window=12, n_subsamples=4)
        out["rv_subsampled_240min"] = _subsampled_rv(out["log_return_5min"], window=48, n_subsamples=4)

    # Keep downstream behavior deterministic: missing values from long windows are zero-filled.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out.loc[:, numeric_cols] = out.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

