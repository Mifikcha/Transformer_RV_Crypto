"""
Regression metrics for multi-horizon forecasting.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _qlike_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    yt = np.clip(y_true.astype(float), eps, None)
    yp = np.clip(y_pred.astype(float), eps, None)
    return float(np.mean(np.log(yp) + yt / yp))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> dict[str, float]:
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays [N, H].")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.shape[1] != len(target_columns):
        raise ValueError(
            f"target_columns len={len(target_columns)} does not match H={y_true.shape[1]}"
        )

    out: dict[str, float] = {}
    mse_vals: list[float] = []
    mae_vals: list[float] = []
    r2_vals: list[float] = []
    da_vals: list[float] = []
    qlike_vals: list[float] = []

    for idx, name in enumerate(target_columns):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        mse = float(mean_squared_error(yt, yp))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        da = float(np.mean(np.sign(yt) == np.sign(yp)))
        qlike = _qlike_loss(yt, yp)
        out[f"mse_{name}"] = mse
        out[f"mae_{name}"] = mae
        out[f"r2_{name}"] = r2
        out[f"da_{name}"] = da
        out[f"qlike_{name}"] = qlike
        mse_vals.append(mse)
        mae_vals.append(mae)
        r2_vals.append(r2)
        da_vals.append(da)
        qlike_vals.append(qlike)

    out["mse_mean"] = float(np.mean(mse_vals))
    out["mae_mean"] = float(np.mean(mae_vals))
    out["r2_mean"] = float(np.mean(r2_vals))
    out["da_mean"] = float(np.mean(da_vals))
    out["qlike_mean"] = float(np.mean(qlike_vals))
    return out
