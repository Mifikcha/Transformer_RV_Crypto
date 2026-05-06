"""HAR-RV baseline for multi-horizon RV forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from utils import (
    RV_TARGET_COLS,
    clip_log_predictions_to_train,
    compute_regression_metrics,
    get_default_data_path,
    get_regression_target_columns,
    log_to_rv,
    load_dataset,
    print_regression_metrics,
    rv_to_log,
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
    return_predictions: bool = False,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)

    rv_d = _resolve_rv_base(df)
    X = _build_har_features(rv_d)
    y = df[tgt_cols].astype(float).to_numpy()

    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []
    pred_parts = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train in log-space for QLIKE-compatible behavior.
        y_train_log = rv_to_log(y_train)
        model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
        model.fit(X_train_s, y_train_log)
        y_pred_log = model.predict(X_test_s)
        y_pred_log = clip_log_predictions_to_train(y_pred_log, y_train_log)
        y_pred = log_to_rv(y_pred_log)

        metrics_per_fold.append(
            compute_regression_metrics(y_true=y_test, y_pred=y_pred, target_columns=tgt_cols)
        )
        if return_predictions:
            part = df.iloc[test_idx][["ts"]].copy() if "ts" in df.columns else df.iloc[test_idx].copy()
            part = part.reset_index(drop=True)
            part["fold_id"] = int(fold_id)
            for i, col in enumerate(tgt_cols):
                part[f"actual_{col}"] = y_test[:, i]
                part[f"pred_{col}"] = y_pred[:, i]
            pred_parts.append(part)

    print_regression_metrics(metrics_per_fold, "HAR-RV (Ridge, log-target)", tgt_cols)
    if not return_predictions:
        return metrics_per_fold
    import pandas as pd
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return {"metrics_per_fold": metrics_per_fold, "predictions_df": pred_df}


if __name__ == "__main__":
    run()
