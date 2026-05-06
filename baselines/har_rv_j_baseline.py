"""HAR-RV-J baseline for multi-horizon RV forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from utils import (
    RV_TARGET_COLS,
    compute_regression_metrics,
    get_default_data_path,
    get_regression_target_columns,
    load_dataset,
    print_regression_metrics,
    walk_forward_split,
)

DAY_BARS = 3
WEEK_BARS = 5 * 288
MONTH_BARS = 22 * 288


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        out[i] = arr[max(0, i - window + 1) : i + 1].mean()
    return out


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        out[i] = arr[max(0, i - window + 1) : i + 1].sum()
    return out


def _resolve_returns(df) -> np.ndarray:
    if "log_return_5min" in df.columns:
        r = df["log_return_5min"].astype(float).to_numpy()
    elif "close_perp" in df.columns:
        close = df["close_perp"].astype(float).to_numpy()
        r = np.zeros_like(close, dtype=float)
        if len(close) > 1:
            r[1:] = np.log(np.clip(close[1:] / np.clip(close[:-1], 1e-12, None), 1e-12, None))
    else:
        raise ValueError("HAR-RV-J baseline requires 'log_return_5min' or 'close_perp'.")
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _build_har_rv_j_features(r: np.ndarray) -> np.ndarray:
    rv_var_d = _rolling_sum(r * r, DAY_BARS)
    rv_d = np.sqrt(np.clip(rv_var_d, 0.0, None))
    rv_w = _rolling_mean(rv_d, WEEK_BARS)
    rv_m = _rolling_mean(rv_d, MONTH_BARS)

    abs_prod = np.abs(r) * np.abs(np.concatenate([[0.0], r[:-1]]))
    bv_d = (np.pi / 2.0) * _rolling_sum(abs_prod, DAY_BARS)
    jump_d = np.clip(rv_var_d - bv_d, 0.0, None)
    jump_w = _rolling_mean(jump_d, WEEK_BARS)
    jump_m = _rolling_mean(jump_d, MONTH_BARS)
    return np.column_stack([rv_d, rv_w, rv_m, jump_d, jump_w, jump_m])


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)

    r = _resolve_returns(df)
    X = _build_har_rv_j_features(r)
    y = df[tgt_cols].astype(float).to_numpy()

    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []
    for train_idx, test_idx in splits:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_pred = np.clip(y_pred, 1e-8, None)

        metrics_per_fold.append(
            compute_regression_metrics(y_true=y_test, y_pred=y_pred, target_columns=tgt_cols)
        )

    print_regression_metrics(metrics_per_fold, "HAR-RV-J (Ridge)", tgt_cols)
    return metrics_per_fold


if __name__ == "__main__":
    run()
