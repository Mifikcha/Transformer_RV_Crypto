"""LightGBM regression baseline (multi-output RV forecasting)."""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from utils import (
    RV_TARGET_COLS,
    compute_regression_metrics,
    get_default_data_path,
    get_feature_columns_recommended_or_all,
    get_regression_target_columns,
    log_to_rv,
    load_dataset,
    print_regression_metrics,
    rv_to_log,
    walk_forward_split,
)


def _qlike_obj_logspace(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Custom LightGBM objective for QLIKE in log-space.

    Loss: L = y_pred + exp(y_true - y_pred)
    grad = 1 - exp(y_true - y_pred)
    hess = exp(y_true - y_pred)
    """
    # Numerical stability: exp() can overflow if (y_true - y_pred) is large.
    # Clipping keeps grads/hessians finite without changing behavior in the
    # typical regime.
    yt = y_true.astype(np.float64, copy=False)
    yp = y_pred.astype(np.float64, copy=False)
    diff = np.clip(yt - yp, -50.0, 50.0)
    e = np.exp(diff)
    grad = 1.0 - e
    hess = e
    return grad, hess


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns_recommended_or_all(df)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)

    X = df[feat_cols].astype(float).fillna(0.0).values
    y = df[tgt_cols].astype(float).values
    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        base = lgb.LGBMRegressor(
            objective=_qlike_obj_logspace,
            max_depth=6,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
        model = MultiOutputRegressor(base)
        y_train_log = rv_to_log(y_train)
        model.fit(X_train_s, y_train_log)
        y_pred_log = model.predict(X_test_s)
        y_pred = log_to_rv(y_pred_log)

        metrics_per_fold.append(
            compute_regression_metrics(y_true=y_test, y_pred=y_pred, target_columns=tgt_cols)
        )

    print_regression_metrics(metrics_per_fold, "LightGBM (QLIKE obj, log-target)", tgt_cols)
    return metrics_per_fold


if __name__ == "__main__":
    run()
