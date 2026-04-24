"""
Group ablation for RV regression: remove each feature family and compare metric deltas.
"""

from __future__ import annotations

import io
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_SCRIPT_DIR)
_BASELINES = os.path.join(_BASE, "baselines")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _BASELINES not in sys.path:
    sys.path.insert(0, _BASELINES)

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_groups import FEATURE_GROUPS
from utils import (
    RV_TARGET_COLS,
    compute_regression_metrics,
    get_default_data_path,
    get_feature_columns,
    load_dataset,
    walk_forward_split,
)


def _regressor_config() -> dict[str, float | int | str]:
    return {
        "objective": "regression",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
    }


def _out(msg: str, log_file: io.TextIOWrapper | None) -> None:
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def _predict_multi_target(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit one regressor per target column and return stacked predictions [N, H]."""
    preds: list[np.ndarray] = []
    for col_idx in range(y_train.shape[1]):
        model = lgb.LGBMRegressor(**_regressor_config())
        model.fit(X_train, y_train[:, col_idx])
        preds.append(model.predict(X_test))
    return np.column_stack(preds)


def _run_with_features(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feat_cols: list[str],
    target_cols: list[str],
) -> dict[str, float]:
    """Train regression models per fold on selected features and aggregate metrics."""
    metrics_per_fold: list[dict[str, float]] = []
    X_sub = X[feat_cols].astype(float).fillna(0.0)

    for train_idx, test_idx in splits:
        X_train_raw = X_sub.iloc[train_idx].values
        X_test_raw = X_sub.iloc[test_idx].values
        y_train = Y.iloc[train_idx].values
        y_test = Y.iloc[test_idx].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw)
        X_test_s = scaler.transform(X_test_raw)

        y_pred = _predict_multi_target(X_train_s, y_train, X_test_s)
        metrics = compute_regression_metrics(y_test, y_pred, target_cols)
        metrics_per_fold.append(metrics)

    keys = ["mse_mean", "mae_mean", "r2_mean", "da_mean", "qlike_mean"]
    return {k: float(np.mean([m[k] for m in metrics_per_fold])) for k in keys}


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    log_file: io.TextIOWrapper | None = None,
) -> pd.DataFrame:
    """Baseline with all features, then ablate each group and compute metric deltas."""
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    target_cols = [c for c in RV_TARGET_COLS if c in df.columns]
    if not target_cols:
        raise ValueError("No RV target columns found in dataset.")

    X = df[feat_cols].astype(float).fillna(0.0)
    Y = df[target_cols].astype(float).fillna(0.0)
    splits = walk_forward_split(df, n_splits=n_splits)

    baseline_metrics = _run_with_features(X, Y, splits, feat_cols, target_cols)

    rows: list[dict[str, float | str]] = []
    for group_name, group_features in FEATURE_GROUPS.items():
        group_cols = set(group_features)
        ablated_cols = [c for c in feat_cols if c not in group_cols]
        if not ablated_cols:
            continue

        ablated_metrics = _run_with_features(X, Y, splits, ablated_cols, target_cols)

        row = {
            "group": group_name,
            "baseline_mse_mean": baseline_metrics["mse_mean"],
            "ablated_mse_mean": ablated_metrics["mse_mean"],
            "delta_mse_mean": ablated_metrics["mse_mean"] - baseline_metrics["mse_mean"],
            "baseline_mae_mean": baseline_metrics["mae_mean"],
            "ablated_mae_mean": ablated_metrics["mae_mean"],
            "delta_mae_mean": ablated_metrics["mae_mean"] - baseline_metrics["mae_mean"],
            "baseline_r2_mean": baseline_metrics["r2_mean"],
            "ablated_r2_mean": ablated_metrics["r2_mean"],
            "delta_r2_mean": baseline_metrics["r2_mean"] - ablated_metrics["r2_mean"],
            "baseline_da_mean": baseline_metrics["da_mean"],
            "ablated_da_mean": ablated_metrics["da_mean"],
            "delta_da_mean": baseline_metrics["da_mean"] - ablated_metrics["da_mean"],
            "baseline_qlike_mean": baseline_metrics["qlike_mean"],
            "ablated_qlike_mean": ablated_metrics["qlike_mean"],
            "delta_qlike_mean": ablated_metrics["qlike_mean"] - baseline_metrics["qlike_mean"],
        }
        rows.append(row)

    result = pd.DataFrame(rows)

    _out("\n" + "=" * 110, log_file)
    _out("  GROUP ABLATION (RV regression, LightGBMRegressor)", log_file)
    _out("=" * 110, log_file)
    _out(
        "  Baseline (all features): "
        f"mse={baseline_metrics['mse_mean']:.6f} "
        f"mae={baseline_metrics['mae_mean']:.6f} "
        f"r2={baseline_metrics['r2_mean']:.4f} "
        f"da={baseline_metrics['da_mean']:.4f} "
        f"qlike={baseline_metrics['qlike_mean']:.6f}",
        log_file,
    )
    _out("-" * 110, log_file)
    _out(
        f"  {'Group':<18}  {'delta_mse':>10}  {'delta_r2':>10}  {'delta_qlike':>12}  {'delta_mae':>10}  {'delta_da':>10}",
        log_file,
    )
    _out("-" * 110, log_file)
    for _, r in result.iterrows():
        _out(
            f"  {r['group']:<18}  {r['delta_mse_mean']:>+10.6f}  {r['delta_r2_mean']:>+10.6f}  {r['delta_qlike_mean']:>+12.6f}  {r['delta_mae_mean']:>+10.6f}  {r['delta_da_mean']:>+10.6f}",
            log_file,
        )
    _out("=" * 110 + "\n", log_file)

    return result
