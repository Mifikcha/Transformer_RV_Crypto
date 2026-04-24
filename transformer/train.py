"""
Training utilities for RV forecasting.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformer.config import MODELS_DIR, ModelConfig, TrainConfig, ensure_output_dirs
from transformer.dataset import (
    TimeSeriesRegressionDataset,
    load_base_dataframe,
    load_recommended_features,
    make_fold_data_regression,
    resolve_features,
)
from transformer.loss import MSELossWrapper, RVLogAwareLoss
from transformer.metrics import compute_regression_metrics
from transformer.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rv_warmup_cosine_lr(
    epoch: int, warmup_epochs: int, max_epochs: int, base_lr: float
) -> float:
    """Linear warmup, then cosine decay to ~0 (RV plan v4 step 2)."""
    w = int(warmup_epochs)
    if w <= 0:
        w = 0
    if epoch < w:
        return float(base_lr) * (epoch + 1) / max(w, 1)
    denom = max(max_epochs - w, 1)
    progress = (epoch - w) / denom
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _model_type_uses_har_context(model_type: str) -> bool:
    return str(model_type).lower().strip() in {"patch_encoder", "encoder_only"}


def _build_fold_dataloaders_regression(
    fold,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    use_har: bool,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TimeSeriesRegressionDataset(
        features_array=fold.X_train,
        targets_array=fold.y_train,
        seq_len=seq_len,
        har_array=fold.har_train if use_har else None,
    )
    ctx = seq_len - 1
    if ctx > 0:
        x_val_full = np.concatenate([fold.X_train[-ctx:], fold.X_val], axis=0)
        y_val_full = np.concatenate([fold.y_train[-ctx:], fold.y_val], axis=0)
        har_val_full = (
            np.concatenate([fold.har_train[-ctx:], fold.har_val], axis=0)
            if use_har
            else None
        )
    else:
        x_val_full = fold.X_val
        y_val_full = fold.y_val
        har_val_full = fold.har_val if use_har else None

    val_ds = TimeSeriesRegressionDataset(
        features_array=x_val_full,
        targets_array=y_val_full,
        seq_len=seq_len,
        har_array=har_val_full,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, val_loader


def _build_fold_dataloaders_with_inner_val(
    fold,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    use_har: bool,
    val_frac: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split X_train into inner_train + inner_val for early stopping, keep X_val as held-out test."""
    n = len(fold.X_train)
    n_inner_val = int(round(n * val_frac))
    if n_inner_val <= 0 or n_inner_val >= n:
        raise ValueError(f"val_frac={val_frac} produced invalid inner-val size (n={n}, n_inner_val={n_inner_val}).")

    X_it = fold.X_train[:-n_inner_val]
    y_it = fold.y_train[:-n_inner_val]
    h_it = fold.har_train[:-n_inner_val] if use_har else None
    X_iv = fold.X_train[-n_inner_val:]
    y_iv = fold.y_train[-n_inner_val:]
    h_iv = fold.har_train[-n_inner_val:] if use_har else None

    ctx = seq_len - 1
    if ctx > 0:
        x_iv_full = np.concatenate([X_it[-ctx:], X_iv], axis=0)
        y_iv_full = np.concatenate([y_it[-ctx:], y_iv], axis=0)
        h_iv_full = np.concatenate([h_it[-ctx:], h_iv], axis=0) if use_har else None
        x_te_full = np.concatenate([fold.X_train[-ctx:], fold.X_val], axis=0)
        y_te_full = np.concatenate([fold.y_train[-ctx:], fold.y_val], axis=0)
        h_te_full = (
            np.concatenate([fold.har_train[-ctx:], fold.har_val], axis=0)
            if use_har
            else None
        )
    else:
        x_iv_full, y_iv_full, h_iv_full = X_iv, y_iv, h_iv
        x_te_full, y_te_full, h_te_full = fold.X_val, fold.y_val, fold.har_val if use_har else None

    train_ds = TimeSeriesRegressionDataset(
        features_array=X_it, targets_array=y_it, seq_len=seq_len, har_array=h_it
    )
    inner_val_ds = TimeSeriesRegressionDataset(
        features_array=x_iv_full, targets_array=y_iv_full, seq_len=seq_len, har_array=h_iv_full
    )
    test_ds = TimeSeriesRegressionDataset(
        features_array=x_te_full, targets_array=y_te_full, seq_len=seq_len, har_array=h_te_full
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    inner_val_loader = DataLoader(inner_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, inner_val_loader, test_loader


def train_walk_forward_regression(
    data_path: str,
    features_path: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    models_dir: str = MODELS_DIR,
    save_models: bool = True,
    model_prefix: str = "fold_reg",
    run_name: str = "train_regression",
    val_frac: float = 0.0,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train transformer in walk-forward fashion.

    Parameters
    ----------
    val_frac : float, default 0.0
        When > 0, split each fold's X_train into inner_train + inner_val (last
        ``val_frac`` fraction). The inner_val is used for early stopping;
        ``X_val`` becomes a clean held-out TEST set used ONLY for reported
        metrics. With ``val_frac=0.0`` keeps legacy behavior (val = test).
    feature_columns : list[str] | None
        If provided, ``features_path`` is ignored and these columns are used
        as features (filtered to numeric present in df).
    """
    ensure_output_dirs()
    os.makedirs(models_dir, exist_ok=True)
    set_seed(train_cfg.seed)
    device = get_device()

    df = load_base_dataframe(data_path)
    if feature_columns is not None:
        import pandas as _pd
        feature_cols = [
            c for c in feature_columns
            if c in df.columns and _pd.api.types.is_numeric_dtype(df[c])
        ]
        if not feature_cols:
            raise ValueError("Provided feature_columns have no numeric overlap with dataset.")
    else:
        rec_features = load_recommended_features(features_path)
        feature_cols = resolve_features(df, rec_features)
    target_cols = [c for c in train_cfg.target_columns if c in df.columns]
    if len(target_cols) != len(train_cfg.target_columns):
        missing = [c for c in train_cfg.target_columns if c not in target_cols]
        raise ValueError(f"Missing regression target columns: {missing}")

    har_mode = str(getattr(model_cfg, "har_mode", "full")).strip().lower()
    if har_mode not in {"full", "none", "weekly_only"}:
        raise ValueError(f"Unsupported har_mode in ModelConfig: {har_mode!r}")

    folds = make_fold_data_regression(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        n_splits=train_cfg.n_splits,
        har_mode=har_mode,
    )
    is_rv_task = str(train_cfg.task).lower() == "rv"
    rv_log_eps = 1e-8
    arch_uses_har = _model_type_uses_har_context(model_cfg.model_type)
    use_har_context = arch_uses_har and har_mode != "none"

    metrics_per_fold: list[dict] = []
    pred_parts: list[pd.DataFrame] = []
    saved_model_paths: list[str] = []

    for fold in folds:
        n_har_context = int(fold.har_train.shape[1]) if use_har_context else 0
        if val_frac and val_frac > 0.0:
            train_loader, val_loader, test_loader = _build_fold_dataloaders_with_inner_val(
                fold=fold,
                seq_len=model_cfg.seq_len,
                batch_size=train_cfg.batch_size,
                num_workers=train_cfg.num_workers,
                use_har=use_har_context,
                val_frac=float(val_frac),
            )
        else:
            train_loader, val_loader = _build_fold_dataloaders_regression(
                fold=fold,
                seq_len=model_cfg.seq_len,
                batch_size=train_cfg.batch_size,
                num_workers=train_cfg.num_workers,
                use_har=use_har_context,
            )
            test_loader = val_loader
        model = build_model(
            model_type=model_cfg.model_type,
            input_dim=len(feature_cols),
            seq_len=model_cfg.seq_len,
            patch_size=model_cfg.patch_size,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers,
            d_ff=model_cfg.d_ff,
            dropout=model_cfg.dropout,
            n_horizons=len(target_cols),
            n_enc_layers=model_cfg.n_enc_layers,
            n_dec_layers=model_cfg.n_dec_layers,
            n_har=n_har_context,
            d_har=16,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        loss_type = str(getattr(train_cfg, "loss_type", "rv_log_aware")).strip().lower()
        if loss_type == "mse":
            criterion = MSELossWrapper()
        elif loss_type == "rv_log_aware":
            if not is_rv_task:
                raise ValueError("loss_type='rv_log_aware' requires task='rv'.")
            criterion = RVLogAwareLoss(alpha=float(getattr(train_cfg, "loss_alpha", 0.7)))
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type!r}")

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(train_cfg.max_epochs):
            lr = _rv_warmup_cosine_lr(
                epoch,
                train_cfg.warmup_epochs,
                train_cfg.max_epochs,
                train_cfg.lr,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            model.train()
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                optimizer.zero_grad(set_to_none=True)
                if use_har_context:
                    har = batch["har"].to(device)
                    pred = model(x, har)
                else:
                    pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                if train_cfg.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=train_cfg.gradient_clip_norm
                    )
                optimizer.step()

            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    if use_har_context:
                        har = batch["har"].to(device)
                        pred = model(x, har)
                    else:
                        pred = model(x)
                    val_losses.append(float(criterion(pred, y).detach().cpu().item()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
                if wait >= train_cfg.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        y_pred_parts: list[np.ndarray] = []
        y_true_parts: list[np.ndarray] = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                y = batch["y"]
                if use_har_context:
                    har = batch["har"].to(device)
                    pred = model(x, har).detach().cpu()
                else:
                    pred = model(x).detach().cpu()
                if is_rv_task:
                    pred = torch.clamp(torch.exp(pred), min=rv_log_eps)
                y_pred_parts.append(pred.numpy())
                y_true_parts.append(y.detach().cpu().numpy())
        y_pred = np.concatenate(y_pred_parts, axis=0) if y_pred_parts else np.empty((0, len(target_cols)))
        y_true = np.concatenate(y_true_parts, axis=0) if y_true_parts else np.empty((0, len(target_cols)))
        metrics = compute_regression_metrics(
            y_true=y_true,
            y_pred=y_pred,
            target_columns=target_cols,
        )
        metrics["fold_id"] = int(fold.fold_id)
        metrics["val_loss"] = float(best_val_loss)
        metrics_per_fold.append(metrics)

        fold_df = df.iloc[fold.val_idx].copy().reset_index(drop=True)
        fold_df["fold_id"] = int(fold.fold_id)
        for i, col in enumerate(target_cols):
            fold_df[f"pred_{col}"] = y_pred[:, i]
            fold_df[f"actual_{col}"] = y_true[:, i]
        pred_parts.append(fold_df)

        if save_models:
            model_path = os.path.join(models_dir, f"{model_prefix}_{fold.fold_id}.pt")
            payload = {
                "state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "effective_n_horizons": len(target_cols),
                "train_config": asdict(train_cfg),
                "feature_columns": feature_cols,
                "target_columns": target_cols,
                "scaler_mean": fold.scaler.mean_.tolist(),
                "scaler_scale": fold.scaler.scale_.tolist(),
                "har_scaler_mean": fold.har_scaler.mean_.tolist() if n_har_context > 0 else [],
                "har_scaler_scale": fold.har_scaler.scale_.tolist() if n_har_context > 0 else [],
                "n_har_context": n_har_context,
                "fold_id": int(fold.fold_id),
                "run_name": run_name,
            }
            torch.save(payload, model_path)
            saved_model_paths.append(model_path)

    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    if not pred_df.empty and "ts" in pred_df.columns:
        pred_df["ts"] = pd.to_datetime(pred_df["ts"], utc=True)
        pred_df = pred_df.sort_values("ts").reset_index(drop=True)

    summary = {
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "metrics_per_fold": metrics_per_fold,
        "saved_model_paths": saved_model_paths,
        "predictions_df": pred_df,
        "device": str(device),
    }
    return summary


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
