"""
Utilities for ablation experiment runners.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from transformer.model import build_model
from transformer.train import train_walk_forward_regression


def _qlike_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    yt = np.clip(y_true.astype(float), eps, None)
    yp = np.clip(y_pred.astype(float), eps, None)
    return float(np.mean(np.log(yp) + yt / yp))


def progress(step: int, total: int, msg: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {msg}")


def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> dict[str, float]:
    """
    Compute per-horizon and mean metrics for ablation experiments.
    Includes: R2, MSE, MAE, QLIKE, bias, corr, overpredict_ratio, p95_abs_err.
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays [N, H].")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.shape[1] != len(target_columns):
        raise ValueError(
            f"target_columns len={len(target_columns)} does not match H={y_true.shape[1]}"
        )

    out: dict[str, float] = {}
    r2_vals: list[float] = []
    mse_vals: list[float] = []
    mae_vals: list[float] = []
    qlike_vals: list[float] = []
    bias_vals: list[float] = []
    corr_vals: list[float] = []
    over_vals: list[float] = []
    p95_vals: list[float] = []

    for idx, name in enumerate(target_columns):
        yt = y_true[:, idx].astype(float)
        yp = y_pred[:, idx].astype(float)
        err = yp - yt

        r2 = float(r2_score(yt, yp))
        mse = float(mean_squared_error(yt, yp))
        mae = float(mean_absolute_error(yt, yp))
        qlike = float(_qlike_loss(yt, yp))
        bias = float(np.mean(err))
        corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float("nan")
        overpredict_ratio = float(np.mean(yp > yt))
        p95_abs_err = float(np.quantile(np.abs(err), 0.95))

        out[f"r2_{name}"] = r2
        out[f"mse_{name}"] = mse
        out[f"mae_{name}"] = mae
        out[f"qlike_{name}"] = qlike
        out[f"bias_{name}"] = bias
        out[f"corr_{name}"] = corr
        out[f"overpredict_ratio_{name}"] = overpredict_ratio
        out[f"p95_abs_err_{name}"] = p95_abs_err

        r2_vals.append(r2)
        mse_vals.append(mse)
        mae_vals.append(mae)
        qlike_vals.append(qlike)
        bias_vals.append(bias)
        corr_vals.append(corr)
        over_vals.append(overpredict_ratio)
        p95_vals.append(p95_abs_err)

    out["r2_mean"] = float(np.mean(r2_vals))
    out["mse_mean"] = float(np.mean(mse_vals))
    out["mae_mean"] = float(np.mean(mae_vals))
    out["qlike_mean"] = float(np.mean(qlike_vals))
    out["bias_mean"] = float(np.mean(bias_vals))
    out["corr_mean"] = float(np.nanmean(corr_vals))
    out["overpredict_ratio_mean"] = float(np.mean(over_vals))
    out["p95_abs_err_mean"] = float(np.mean(p95_vals))
    return out


def _count_model_params(model_cfg, input_dim: int, n_horizons: int) -> int:
    model = build_model(
        model_type=model_cfg.model_type,
        input_dim=input_dim,
        seq_len=model_cfg.seq_len,
        patch_size=model_cfg.patch_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        n_horizons=n_horizons,
        n_enc_layers=model_cfg.n_enc_layers,
        n_dec_layers=model_cfg.n_dec_layers,
        n_har=(6 if str(model_cfg.har_mode).lower() == "full" else 3 if str(model_cfg.har_mode).lower() == "weekly_only" else 0),
        d_har=16,
    )
    n_params = int(sum(p.numel() for p in model.parameters()))
    del model
    return n_params


def run_single_experiment(
    model_cfg,
    train_cfg,
    data_path: str,
    features_path: str,
    experiment_id: str,
    variant_name: str,
    feature_cols_override: list[str] | None = None,
    val_frac: float = 0.0,
) -> dict[str, Any]:
    """
    Run one ablation experiment and return a flat row with extended metrics.
    """
    started = time.time()
    result = train_walk_forward_regression(
        data_path=data_path,
        features_path=features_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        save_models=False,
        model_prefix=f"{experiment_id}_fold",
        run_name=experiment_id,
        val_frac=val_frac,
        feature_columns=feature_cols_override,
    )
    pred_df: pd.DataFrame = result["predictions_df"]
    target_columns: list[str] = list(result["target_columns"])
    feature_columns: list[str] = list(result["feature_columns"])
    if pred_df.empty:
        raise RuntimeError(f"{experiment_id}: empty predictions dataframe.")

    y_true = pred_df[[f"actual_{c}" for c in target_columns]].to_numpy(dtype=float)
    y_pred = pred_df[[f"pred_{c}" for c in target_columns]].to_numpy(dtype=float)
    ext = compute_extended_metrics(y_true=y_true, y_pred=y_pred, target_columns=target_columns)

    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "variant_name": variant_name,
        "model_type": model_cfg.model_type,
        "har_mode": getattr(model_cfg, "har_mode", "full"),
        "loss_type": getattr(train_cfg, "loss_type", "rv_log_aware"),
        "loss_alpha": getattr(train_cfg, "loss_alpha", 0.7),
        "n_folds": int(train_cfg.n_splits),
        "elapsed_sec": round(time.time() - started, 2),
        "n_features": int(len(feature_columns)),
        "n_params": _count_model_params(model_cfg, input_dim=len(feature_columns), n_horizons=len(target_columns)),
    }
    row.update(ext)
    row["model_config"] = asdict(model_cfg)
    row["train_config"] = asdict(train_cfg)
    return row


def save_experiment_csv(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_experiment_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
