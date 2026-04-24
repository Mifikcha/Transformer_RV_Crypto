"""
Shared utilities for baselines:
- RV regression helpers (multi-horizon)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Columns to exclude from features (targets, leakage, aux)
EXCLUDED_COLUMNS = {
    "ts",
    "target_return",
    "target_class",
    "is_valid_target",
    "future_close_30min",
    "future_close_60min",
    "future_close_120min",
    "future_close_240min",
    "delta_log_30min",
    "delta_log_60min",
    "delta_log_120min",
    "delta_log_240min",
    "delta_log_1bar_fwd",
    "delta_log_3bar_fwd",
    "delta_log_12bar_fwd",
    "delta_log_48bar_fwd",
    "delta_log_288bar_fwd",
    "rv_3bar_fwd",
    "rv_12bar_fwd",
    "rv_48bar_fwd",
    "rv_288bar_fwd",
    "base_regression",
    "base_class",
    "trading_class_optimistic",
    "trading_class_base",
    "trading_class_pessimistic",
}

VALID_TARGET_COL = "is_valid_target"

RV_TARGET_COLS = ("rv_3bar_fwd", "rv_12bar_fwd", "rv_48bar_fwd", "rv_288bar_fwd")

def get_default_data_path() -> str:
    """Resolve default dataset path with backward-compatible fallbacks.

    Resolution order:
    1) env var ``DATA_PATH`` (if points to existing file)
    2) common modern path(s)
    3) legacy path(s)
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.environ.get("DATA_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    candidates = [
        os.path.join(base, "target", "btcusdt_5m_final_with_targets.csv"),
        os.path.join(base, "dataset", "get_data", "output", "_main", "btcusdt_5m_final_with_targets.csv"),
        os.path.join(base, "target", "form_target", "btcusdt_5m_final_with_targets.csv"),
        os.path.join(base, "2. Target", "form_target", "btcusdt_5m_final_with_targets.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Keep deterministic default if nothing exists yet.
    return candidates[0]


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV, filter is_valid_target==1 when available, sort by ts."""
    df = pd.read_csv(path)
    if VALID_TARGET_COL in df.columns:
        df = df.loc[df[VALID_TARGET_COL].astype(int) == 1].copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (exclude targets and leakage)."""
    out: list[str] = []
    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def get_regression_target_columns(
    df: pd.DataFrame,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
) -> list[str]:
    cols = [c for c in target_columns if c in df.columns]
    if len(cols) != len(target_columns):
        missing = [c for c in target_columns if c not in cols]
        raise ValueError(f"Missing target columns in dataset: {missing}")
    return cols


def walk_forward_split(
    df: pd.DataFrame, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward: for each fold, train on past chunks, test on next chunk.
    Returns list of (train_idx, test_idx) arrays.
    """
    n = len(df)
    if n < n_splits + 1:
        raise ValueError("Not enough rows for walk-forward splits")
    segment_size = n // (n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_end = (k + 1) * segment_size
        test_end = (k + 2) * segment_size if (k + 2) <= n_splits else n
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(test_end, n))
        if len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> dict[str, float]:
    """
    Regression metrics per horizon and their macro averages.
    Expects shape [N, H] for y_true and y_pred.
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("compute_regression_metrics expects 2D arrays [N, H].")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.shape[1] != len(target_columns):
        raise ValueError(
            f"target_columns length ({len(target_columns)}) "
            f"must match y shape second dim ({y_true.shape[1]})."
        )

    out: dict[str, float] = {}
    mse_list: list[float] = []
    mae_list: list[float] = []
    r2_list: list[float] = []
    da_list: list[float] = []
    qlike_list: list[float] = []

    for idx, col in enumerate(target_columns):
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]
        mse_v = float(mean_squared_error(y_t, y_p))
        mae_v = float(mean_absolute_error(y_t, y_p))
        r2_v = float(r2_score(y_t, y_p))
        da_v = float(np.mean(np.sign(y_t) == np.sign(y_p)))
        eps = 1e-12
        yt_pos = np.clip(y_t, eps, None)
        yp_pos = np.clip(y_p, eps, None)
        qlike_v = float(np.mean(np.log(yp_pos) + yt_pos / yp_pos))
        out[f"mse_{col}"] = mse_v
        out[f"mae_{col}"] = mae_v
        out[f"r2_{col}"] = r2_v
        out[f"da_{col}"] = da_v
        out[f"qlike_{col}"] = qlike_v
        mse_list.append(mse_v)
        mae_list.append(mae_v)
        r2_list.append(r2_v)
        da_list.append(da_v)
        qlike_list.append(qlike_v)

    out["mse_mean"] = float(np.mean(mse_list))
    out["mae_mean"] = float(np.mean(mae_list))
    out["r2_mean"] = float(np.mean(r2_list))
    out["da_mean"] = float(np.mean(da_list))
    out["qlike_mean"] = float(np.mean(qlike_list))
    return out


def print_regression_metrics(
    metrics_per_fold: list[dict[str, float]],
    model_name: str,
    target_columns: list[str],
) -> None:
    """Print regression metrics aggregated over folds (mean +- std)."""
    if not metrics_per_fold:
        print(f"[{model_name}] No folds.")
        return

    print("\n" + "=" * 60)
    print(f"  {model_name}")
    print("=" * 60)
    for metric_name in ("mse", "mae", "r2", "da"):
        mean_key = f"{metric_name}_mean"
        vals = [float(m[mean_key]) for m in metrics_per_fold if mean_key in m]
        if vals:
            print(
                f"  {metric_name.upper():<4} mean: {np.mean(vals):.6f} "
                f"(+- {np.std(vals):.6f})"
            )
    print("  --- Per-horizon (mean over folds) ---")
    for col in target_columns:
        mse_vals = [float(m[f"mse_{col}"]) for m in metrics_per_fold if f"mse_{col}" in m]
        mae_vals = [float(m[f"mae_{col}"]) for m in metrics_per_fold if f"mae_{col}" in m]
        r2_vals = [float(m[f"r2_{col}"]) for m in metrics_per_fold if f"r2_{col}" in m]
        da_vals = [float(m[f"da_{col}"]) for m in metrics_per_fold if f"da_{col}" in m]
        if not mse_vals:
            continue
        print(
            f"    {col}: MSE {np.mean(mse_vals):.6f}, MAE {np.mean(mae_vals):.6f}, "
            f"R2 {np.mean(r2_vals):.4f}, DA {np.mean(da_vals):.4f}, "
            f"QLIKE {np.mean([float(m[f'qlike_{col}']) for m in metrics_per_fold if f'qlike_{col}' in m]):.6f}"
        )
    print("=" * 60 + "\n")
