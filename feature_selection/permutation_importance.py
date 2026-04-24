"""
Permutation importance on walk-forward validation windows for RV regression.
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
from sklearn.inspection import permutation_importance
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
    n_repeats: int = 5,
    log_file: io.TextIOWrapper | None = None,
) -> pd.DataFrame:
    """Compute permutation importance per fold and RV target, aggregate mean/std."""
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    target_cols = [c for c in RV_TARGET_COLS if c in df.columns]
    if not target_cols:
        raise ValueError("No RV target columns found in dataset.")

    X = df[feat_cols].astype(float).fillna(0.0)
    Y = df[target_cols].astype(float).fillna(0.0)

    splits = walk_forward_split(df, n_splits=n_splits)
    perm_importances: list[dict[str, float]] = []

    for train_idx, test_idx in splits:
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        per_target_scores: list[np.ndarray] = []
        for target_col in target_cols:
            y_train = Y[target_col].iloc[train_idx].values
            y_test = Y[target_col].iloc[test_idx].values

            model = lgb.LGBMRegressor(**_regressor_config())
            model.fit(X_train_s, y_train)

            pi = permutation_importance(
                model,
                X_test_s,
                y_test,
                n_repeats=n_repeats,
                random_state=42,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            per_target_scores.append(pi.importances_mean)

        fold_importance = np.mean(per_target_scores, axis=0)
        perm_importances.append(dict(zip(feat_cols, fold_importance)))

    rows = []
    for col in feat_cols:
        vals = [d[col] for d in perm_importances]
        rows.append(
            {
                "feature": col,
                "group": get_group_for_feature(col),
                "mean_perm_importance": float(np.mean(vals)),
                "std_perm_importance": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("mean_perm_importance", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    _out("\n" + "=" * 70, log_file)
    _out("  PERMUTATION IMPORTANCE (neg_MSE score drop, averaged over RV targets)", log_file)
    _out("=" * 70, log_file)
    top = result.head(30)
    _out("\n  Top 30 features:", log_file)
    for _, r in top.iterrows():
        _out(
            f"    {r['rank']:3d}  {r['feature']:<30}  {r['mean_perm_importance']:>10.4f}  (+- {r['std_perm_importance']:.4f})  [{r['group']}]",
            log_file,
        )
    group_sum = result.groupby("group", sort=False)["mean_perm_importance"].sum().sort_values(ascending=False)
    _out("\n  Sum of permutation importance by group:", log_file)
    for grp, s in group_sum.items():
        _out(f"    {grp:<20}  {s:>10.4f}", log_file)
    _out("=" * 70 + "\n", log_file)

    return result
