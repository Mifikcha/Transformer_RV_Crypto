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


def dm_bootstrap_compare_pred_dfs(
    *,
    pred_m0: pd.DataFrame,
    pred_m1: pd.DataFrame,
    target_columns: list[str],
    block_size: int = 48,
    n_boot: int = 2000,
    hac_lags: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Compute DM-test + block-bootstrap CI between two pred_df streams.

    pred_df format: must contain columns actual_<target>, pred_<target>, and optionally ts.
    Returns keys dm_p_<target>/dm_stat_<target>/diff_mean_<target>/diff_ci_* and aggregate (suffix 'agg').
    """
    from scripts.stats_tests import dm_test, moving_block_bootstrap_mean_ci, qlike_series

    # Align on timestamps if present.
    if "ts" in pred_m0.columns and "ts" in pred_m1.columns:
        a = pred_m0.copy()
        b = pred_m1.copy()
        a["ts"] = pd.to_datetime(a["ts"], utc=True, errors="coerce")
        b["ts"] = pd.to_datetime(b["ts"], utc=True, errors="coerce")
        merged = a.merge(b, on="ts", how="inner", suffixes=("_m0", "_m1"))
        m0 = pd.DataFrame({"ts": merged["ts"]})
        m1 = pd.DataFrame({"ts": merged["ts"]})
        for col in target_columns:
            m0[f"actual_{col}"] = merged[f"actual_{col}_m0"]
            m0[f"pred_{col}"] = merged[f"pred_{col}_m0"]
            m1[f"actual_{col}"] = merged[f"actual_{col}_m1"]
            m1[f"pred_{col}"] = merged[f"pred_{col}_m1"]
    else:
        m0, m1 = pred_m0, pred_m1

    rng = np.random.default_rng(int(seed))
    out: dict[str, float] = {}
    losses0 = []
    losses1 = []
    for col in target_columns:
        l0 = qlike_series(m0[f"actual_{col}"].to_numpy(), m0[f"pred_{col}"].to_numpy())
        l1 = qlike_series(m1[f"actual_{col}"].to_numpy(), m1[f"pred_{col}"].to_numpy())
        losses0.append(l0.reshape(-1, 1))
        losses1.append(l1.reshape(-1, 1))

        dm = dm_test(l0, l1, hac_lags=hac_lags, alternative="two-sided")
        d = l0 - l1
        mean_d, lo, hi = moving_block_bootstrap_mean_ci(
            d, block_size=int(block_size), n_boot=int(n_boot), alpha=0.05, rng=rng
        )
        out[f"dm_stat_{col}"] = float(dm.dm_stat)
        out[f"dm_p_{col}"] = float(dm.p_value)
        out[f"diff_mean_{col}"] = float(mean_d)
        out[f"diff_ci_lo_{col}"] = float(lo)
        out[f"diff_ci_hi_{col}"] = float(hi)

    if losses0:
        agg0 = np.mean(np.concatenate(losses0, axis=1), axis=1)
        agg1 = np.mean(np.concatenate(losses1, axis=1), axis=1)
        dm = dm_test(agg0, agg1, hac_lags=hac_lags, alternative="two-sided")
        d = agg0 - agg1
        mean_d, lo, hi = moving_block_bootstrap_mean_ci(
            d, block_size=int(block_size), n_boot=int(n_boot), alpha=0.05, rng=rng
        )
        out["dm_stat_agg"] = float(dm.dm_stat)
        out["dm_p_agg"] = float(dm.p_value)
        out["diff_mean_agg"] = float(mean_d)
        out["diff_ci_lo_agg"] = float(lo)
        out["diff_ci_hi_agg"] = float(hi)

    return out


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
