"""LightGBM regression baseline (multi-output RV forecasting)."""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from utils import (
    RV_TARGET_COLS,
    compute_regression_metrics,
    get_default_data_path,
    get_feature_columns_recommended_or_all,
    get_regression_target_columns,
    clip_log_predictions_to_train,
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
    # LightGBM can crash when Hessians become (near-)zero for many rows.
    # Keep them strictly positive and not too extreme.
    e = np.clip(e, 1e-3, 1e3)
    grad = 1.0 - e
    hess = e
    return grad, hess


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
    return_predictions: bool = False,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns_recommended_or_all(df)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)

    X = df[feat_cols].astype(float).fillna(0.0).values
    y = df[tgt_cols].astype(float).values
    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []
    pred_parts = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        y_train_log = rv_to_log(y_train)
        y_pred_log_parts: list[np.ndarray] = []
        for i in range(y_train_log.shape[1]):
            # For the custom QLIKE objective in log-space, the optimal constant
            # predictor is c = log(mean(exp(y))) = log(mean(RV)).
            # LightGBM's default 'boost_from_average' logic is tuned for L2 and
            # can start from mean(log RV), which underestimates RV by Jensen.
            init_c = float(np.log(np.mean(np.clip(y_train[:, i].astype(float), 1e-12, None))))
            init_score = np.full(shape=(X_train_s.shape[0],), fill_value=init_c, dtype=np.float64)

            est = lgb.LGBMRegressor(
                objective=_qlike_obj_logspace,
                boost_from_average=False,
                max_depth=6,
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
            est.fit(X_train_s, y_train_log[:, i], init_score=init_score)

            pred_tr = np.asarray(est.predict(X_train_s), dtype=float)
            bias = float(np.mean(y_train_log[:, i].astype(float) - pred_tr))

            pred_te = np.asarray(est.predict(X_test_s), dtype=float) + bias
            pred_te = clip_log_predictions_to_train(pred_te, y_train_log[:, i])
            y_pred_log_parts.append(pred_te.reshape(-1, 1))

        y_pred_log = np.concatenate(y_pred_log_parts, axis=1)
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

    print_regression_metrics(metrics_per_fold, "LightGBM (QLIKE obj, log-target)", tgt_cols)
    if not return_predictions:
        return metrics_per_fold
    import pandas as pd
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return {"metrics_per_fold": metrics_per_fold, "predictions_df": pred_df}


if __name__ == "__main__":
    run()
