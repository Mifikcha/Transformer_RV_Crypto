"""HAR-RV baseline for multi-horizon RV forecasting."""

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

WEEK_BARS = 5 * 288
MONTH_BARS = 22 * 288


def _resolve_rv_base(df) -> np.ndarray:
    """Pick the shortest-horizon realized-volatility proxy available."""
    candidates = [
        "rv_gk_15min",
        "realized_vol_15min",
        "rv_parkinson_15min",
        "rv_rs_15min",
    ]
    for col in candidates:
        if col in df.columns:
            return df[col].astype(float).to_numpy()
    raise ValueError(
        "HAR-RV baseline requires at least one short-horizon RV feature: "
        "rv_gk_15min | realized_vol_15min | rv_parkinson_15min | rv_rs_15min."
    )


def _build_har_features(rv_d: np.ndarray) -> np.ndarray:
    rv_d = np.nan_to_num(rv_d.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    rv_w = np.empty_like(rv_d, dtype=float)
    rv_m = np.empty_like(rv_d, dtype=float)
    for i in range(len(rv_d)):
        rv_w[i] = rv_d[max(0, i - WEEK_BARS + 1) : i + 1].mean()
        rv_m[i] = rv_d[max(0, i - MONTH_BARS + 1) : i + 1].mean()
    return np.column_stack([rv_d, rv_w, rv_m])


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)

    rv_d = _resolve_rv_base(df)
    X = _build_har_features(rv_d)
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

    print_regression_metrics(metrics_per_fold, "HAR-RV (Ridge)", tgt_cols)
    return metrics_per_fold


if __name__ == "__main__":
    run()
