"""
Feature importance (gain) from LightGBM regressor over walk-forward folds.
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

from feature_groups import get_group_for_feature
from utils import (
    RV_TARGET_COLS,
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
        "importance_type": "gain",
    }


def _out(msg: str, log_file: io.TextIOWrapper | None) -> None:
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    log_file: io.TextIOWrapper | None = None,
) -> pd.DataFrame:
    """Train per fold and per RV target, aggregate gain importance."""
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    target_cols = [c for c in RV_TARGET_COLS if c in df.columns]
    if not target_cols:
        raise ValueError("No RV target columns found in dataset.")

    X = df[feat_cols].astype(float).fillna(0.0)
    Y = df[target_cols].astype(float).fillna(0.0)

    splits = walk_forward_split(df, n_splits=n_splits)
    importances_per_fold: list[dict[str, float]] = []

    for train_idx, _ in splits:
        X_train = X.iloc[train_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        per_target_importances: list[np.ndarray] = []
        for target_col in target_cols:
            y_train = Y[target_col].iloc[train_idx].values
            model = lgb.LGBMRegressor(**_regressor_config())
            model.fit(X_train_s, y_train)
            per_target_importances.append(model.feature_importances_.astype(float))

        fold_importance = np.mean(per_target_importances, axis=0)
        importances_per_fold.append(dict(zip(feat_cols, fold_importance)))

    rows = []
    for col in feat_cols:
        vals = [d[col] for d in importances_per_fold]
        rows.append(
            {
                "feature": col,
                "group": get_group_for_feature(col),
                "mean_importance": float(np.mean(vals)),
                "std_importance": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    _out("\n" + "=" * 70, log_file)
    _out("  BUILTIN IMPORTANCE (LightGBMRegressor gain, averaged over RV targets)", log_file)
    _out("=" * 70, log_file)
    top = result.head(30)
    _out("\n  Top 30 features:", log_file)
    for _, r in top.iterrows():
        _out(
            f"    {r['rank']:3d}  {r['feature']:<30}  {r['mean_importance']:>10.2f}  (+- {r['std_importance']:.2f})  [{r['group']}]",
            log_file,
        )
    group_sum = result.groupby("group", sort=False)["mean_importance"].sum().sort_values(ascending=False)
    _out("\n  Sum of importance by group:", log_file)
    for grp, s in group_sum.items():
        _out(f"    {grp:<20}  {s:>10.2f}", log_file)
    _out("=" * 70 + "\n", log_file)

    return result
