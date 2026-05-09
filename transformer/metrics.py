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
    hmse_vals: list[float] = []
    bias_vals: list[float] = []
    corr_vals: list[float] = []
    p95_vals: list[float] = []

    for idx, name in enumerate(target_columns):
        yt = y_true[:, idx].astype(float)
        yp = y_pred[:, idx].astype(float)
        err = yp - yt
        mse = float(mean_squared_error(yt, yp))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        # Directional Accuracy for RV must be about *change direction*, not sign(RV):
        # RV is strictly positive, so sign(yt)==sign(yp) degenerates to ~1.0.
        # We compare the sign of 1-step changes in log-space: sign(Δlog yt) == sign(Δlog yp).
        eps = 1e-12
        yt_pos = np.clip(yt.astype(float), eps, None)
        yp_pos = np.clip(yp.astype(float), eps, None)
        if len(yt_pos) >= 2:
            dyt = np.diff(np.log(yt_pos))
            dyp = np.diff(np.log(yp_pos))
            da = float(np.mean(np.sign(dyt) == np.sign(dyp)))
        else:
            da = float("nan")
        qlike = _qlike_loss(yt, yp)
        hmse = float(np.mean(((yp.astype(float) - yt_pos) / yt_pos) ** 2))
        bias = float(np.mean(err))
        corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float("nan")
        p95_abs_err = float(np.quantile(np.abs(err), 0.95))
        out[f"mse_{name}"] = mse
        out[f"mae_{name}"] = mae
        out[f"r2_{name}"] = r2
        out[f"da_{name}"] = da
        out[f"qlike_{name}"] = qlike
        out[f"hmse_{name}"] = hmse
        out[f"bias_{name}"] = bias
        out[f"corr_{name}"] = corr
        out[f"p95_abs_err_{name}"] = p95_abs_err
        mse_vals.append(mse)
        mae_vals.append(mae)
        r2_vals.append(r2)
        da_vals.append(da)
        qlike_vals.append(qlike)
        hmse_vals.append(hmse)
        bias_vals.append(bias)
        corr_vals.append(corr)
        p95_vals.append(p95_abs_err)

    out["mse_mean"] = float(np.mean(mse_vals))
    out["mae_mean"] = float(np.mean(mae_vals))
    out["r2_mean"] = float(np.mean(r2_vals))
    out["da_mean"] = float(np.mean(da_vals))
    out["qlike_mean"] = float(np.mean(qlike_vals))
    out["hmse_mean"] = float(np.mean(hmse_vals))
    out["bias_mean"] = float(np.mean(bias_vals))
    out["corr_mean"] = float(np.nanmean(corr_vals))
    out["p95_abs_err_mean"] = float(np.mean(p95_vals))
    return out
