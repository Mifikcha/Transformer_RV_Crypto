"""
RV-only orchestrator for Transformer training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformer.config import (
    AppConfig,
    ModelConfig,
    PICTURES_DIR,
    PREDICTIONS_DIR,
    TrainConfig,
    ensure_output_dirs,
)
from transformer import train as _train_mod
from transformer.train import train_walk_forward_regression

LOG_PATH = os.path.join(PROJECT_ROOT, "log_tranformer")
LOGGER = logging.getLogger("run_transformer")


def _setup_logger() -> None:
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def _emit(msg: str) -> None:
    print(msg)
    LOGGER.info(msg)


def _emit_json(title: str, payload: Any) -> None:
    _emit(f"{title}: {json.dumps(payload, ensure_ascii=False, default=str)}")


def _emit_device_diagnostics(requested: str) -> None:
    _emit("=" * 80)
    _emit("DEVICE DIAGNOSTICS")
    _emit("=" * 80)
    _emit(f"requested_device        : {requested}")
    _emit(f"python_executable       : {sys.executable}")
    _emit(f"python_version          : {sys.version.split()[0]}")
    _emit(f"torch_version           : {torch.__version__}")
    _emit(f"torch_cuda_build        : {torch.version.cuda}")
    _emit(f"cudnn_version           : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}")
    _emit(f"cuda_available          : {torch.cuda.is_available()}")
    _emit(f"cuda_device_count       : {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    _emit(f"CUDA_VISIBLE_DEVICES    : {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            _emit(f"  gpu[{i}]: {p.name} | mem={p.total_memory / (1024 ** 3):.1f}GB | cc={p.major}.{p.minor}")
    _emit("=" * 80)


def _configure_device(requested: str) -> torch.device:
    """Validate/force device choice. Fail-fast when --device cuda but CUDA missing."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested, but torch.cuda.is_available() is False. "
            "Check torch build (torch.version.cuda must not be None), driver, and CUDA_VISIBLE_DEVICES. "
            "See DEVICE DIAGNOSTICS above."
        )
    if requested == "cpu":
        forced = torch.device("cpu")
        _train_mod.get_device = lambda: forced  # override train.get_device for this run
        return forced
    # auto: keep existing behavior (cuda if available, else cpu).
    use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda"}
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    return device


def _metric_mean_std(metrics_per_fold: list[dict], key: str) -> tuple[float, float]:
    vals = [float(m[key]) for m in metrics_per_fold if key in m and np.isfinite(m[key])]
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


def _print_rv_metrics(metrics_per_fold: list[dict], target_columns: list[str]) -> None:
    _emit("\n" + "=" * 80)
    _emit("RV METRICS (walk-forward)")
    _emit("=" * 80)
    for key in ("mse_mean", "mae_mean", "r2_mean", "da_mean", "qlike_mean"):
        mean_v, std_v = _metric_mean_std(metrics_per_fold, key)
        _emit(f"{key:20s}: {mean_v:>10.6f} (+- {std_v:.6f})")
    _emit("-" * 80)
    for col in target_columns:
        mse_m, _ = _metric_mean_std(metrics_per_fold, f"mse_{col}")
        mae_m, _ = _metric_mean_std(metrics_per_fold, f"mae_{col}")
        r2_m, _ = _metric_mean_std(metrics_per_fold, f"r2_{col}")
        qlike_m, _ = _metric_mean_std(metrics_per_fold, f"qlike_{col}")
        _emit(f"{col:20s}: MSE={mse_m:.6f} MAE={mae_m:.6f} R2={r2_m:.4f} QLIKE={qlike_m:.6f}")
    _emit("=" * 80)
    _emit_json("RV metrics per fold", metrics_per_fold)


def _print_feature_summary(feature_columns: list[str], preview: int = 20) -> None:
    _emit("\n" + "=" * 80)
    _emit("FEATURE SET SUMMARY")
    _emit("=" * 80)
    _emit(f"feature_count          : {len(feature_columns)}")
    if not feature_columns:
        _emit("feature_preview        : []")
    else:
        head = feature_columns[:preview]
        tail_count = max(0, len(feature_columns) - len(head))
        _emit(f"feature_preview        : {head}")
        if tail_count > 0:
            _emit(f"feature_preview_tail   : +{tail_count} more features")
    _emit("=" * 80)


def _print_rv_fold_diagnostics(metrics_per_fold: list[dict], target_columns: list[str]) -> None:
    if not metrics_per_fold:
        return
    _emit("\n" + "=" * 80)
    _emit("RV FOLD DIAGNOSTICS")
    _emit("=" * 80)
    fold_df = pd.DataFrame(metrics_per_fold).sort_values("fold_id")
    if "r2_mean" in fold_df.columns:
        best_idx = int(fold_df["r2_mean"].idxmax())
        worst_idx = int(fold_df["r2_mean"].idxmin())
        best = fold_df.loc[best_idx]
        worst = fold_df.loc[worst_idx]
        _emit(
            f"best_fold_by_r2        : fold={int(best['fold_id'])} "
            f"r2_mean={float(best['r2_mean']):.4f} mse_mean={float(best['mse_mean']):.6f}"
        )
        _emit(
            f"worst_fold_by_r2       : fold={int(worst['fold_id'])} "
            f"r2_mean={float(worst['r2_mean']):.4f} mse_mean={float(worst['mse_mean']):.6f}"
        )
    if "val_loss" in fold_df.columns:
        _emit(
            f"val_loss_range          : min={float(fold_df['val_loss'].min()):.6f} "
            f"max={float(fold_df['val_loss'].max()):.6f}"
        )
    _emit(
        f"r2_mean_cv             : {float(fold_df['r2_mean'].std() / (abs(fold_df['r2_mean'].mean()) + 1e-12)):.4f}"
    )
    _emit("-" * 80)
    for col in target_columns:
        if f"r2_{col}" in fold_df.columns:
            _emit(
                f"{col:20s}: "
                f"r2_min={float(fold_df[f'r2_{col}'].min()):.4f} "
                f"r2_max={float(fold_df[f'r2_{col}'].max()):.4f} "
                f"r2_std={float(fold_df[f'r2_{col}'].std()):.4f}"
            )
    _emit("=" * 80)


def _print_prediction_diagnostics(pred_df: pd.DataFrame, target_columns: list[str]) -> None:
    if pred_df.empty:
        return
    _emit("\n" + "=" * 80)
    _emit("PREDICTION DIAGNOSTICS")
    _emit("=" * 80)
    for col in target_columns:
        act_col = f"actual_{col}"
        pred_col = f"pred_{col}"
        if act_col not in pred_df.columns or pred_col not in pred_df.columns:
            continue
        tmp = pred_df[[act_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if tmp.empty:
            continue
        err = (tmp[pred_col] - tmp[act_col]).astype(float)
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        corr = float(np.corrcoef(tmp[act_col].values, tmp[pred_col].values)[0, 1]) if len(tmp) > 1 else float("nan")
        p95_abs = float(np.quantile(np.abs(err.values), 0.95))
        _emit(
            f"{col:20s}: mae={mae:.6f} bias={bias:+.6f} corr={corr:.4f} p95_abs_err={p95_abs:.6f}"
        )
    _emit("-" * 80)
    if "ts" in pred_df.columns:
        last_ts = pd.to_datetime(pred_df["ts"], utc=True, errors="coerce").max()
        _emit(f"latest_prediction_ts   : {last_ts}")
    _emit("=" * 80)


def _save_rv_visualizations(pred_df: pd.DataFrame, bars: int = 1000) -> None:
    os.makedirs(PICTURES_DIR, exist_ok=True)
    if pred_df.empty:
        _emit("[RV] Skip visualization: prediction dataframe is empty.")
        return

    required_base_cols = {"ts", "close_perp"}
    if any(c not in pred_df.columns for c in required_base_cols):
        _emit(f"[RV] Skip visualization: missing required columns: {sorted(required_base_cols)}")
        return

    horizon_specs = [
        ("15m", "15min", "rv_3bar_fwd"),
        ("1h", "1h", "rv_12bar_fwd"),
        ("4h", "4h", "rv_48bar_fwd"),
        ("24h", "24h", "rv_288bar_fwd"),
    ]

    for tf_label, tf_freq, rv_col in horizon_specs:
        act_col = f"actual_{rv_col}"
        pred_col = f"pred_{rv_col}"
        need = ["ts", "close_perp", act_col, pred_col]
        missing = [c for c in need if c not in pred_df.columns]
        if missing:
            _emit(f"[RV] Skip {tf_label} plot: missing columns {missing}")
            continue

        plot_df = pred_df[need].copy()
        plot_df["ts"] = pd.to_datetime(plot_df["ts"], utc=True, errors="coerce")
        plot_df = plot_df.dropna(subset=["ts"])
        if plot_df.empty:
            _emit(f"[RV] Skip {tf_label} plot: no valid timestamps.")
            continue
        plot_df = plot_df.set_index("ts").sort_index()
        plot_df = plot_df.resample(tf_freq).last().dropna()
        if plot_df.empty:
            _emit(f"[RV] Skip {tf_label} plot: no rows after {tf_freq} resample.")
            continue
        plot_df = plot_df.tail(bars)

        fig, ax_price = plt.subplots(figsize=(14, 6), facecolor="black")
        ax_price.set_facecolor("black")
        ax_rv = ax_price.twinx()
        ax_rv.set_facecolor("black")

        ax_price.plot(plot_df.index, plot_df["close_perp"], color="white", lw=1.2, label="price")
        ax_rv.plot(plot_df.index, plot_df[act_col], color="red", lw=1.2, label="actual_RV")
        ax_rv.plot(plot_df.index, plot_df[pred_col], color="#ff69b4", lw=1.2, label="pred_RV")

        ax_price.set_title(
            f"Transformer RV | {tf_label} horizon | last {len(plot_df)} bars",
            color="white",
        )
        ax_price.set_xlabel("timestamp", color="white")
        ax_price.set_ylabel("price", color="white")
        ax_rv.set_ylabel("RV", color="white")
        ax_price.grid(color="#444444", alpha=0.35)

        ax_price.tick_params(colors="white")
        ax_rv.tick_params(colors="white")
        for spine in ax_price.spines.values():
            spine.set_color("white")
        for spine in ax_rv.spines.values():
            spine.set_color("white")

        lines1, labels1 = ax_price.get_legend_handles_labels()
        lines2, labels2 = ax_rv.get_legend_handles_labels()
        legend = ax_rv.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            frameon=True,
            facecolor="black",
            edgecolor="white",
        )
        for text in legend.get_texts():
            text.set_color("white")

        out_name = f"transformer_rv_{tf_label}_last{bars}_bars.png"
        out_path = os.path.join(PICTURES_DIR, out_name)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        _emit(f"[RV] Saved picture: {out_path}")


def run_train_mode(app_cfg: AppConfig) -> None:
    _emit(f"[RV] Run mode: train-rv | log file: {LOG_PATH}")
    _emit_json("App config", asdict(app_cfg))
    ensure_output_dirs()
    # Save "last run" weights into project-level /model directory
    last_run_models_dir = os.path.join(PROJECT_ROOT, "model")
    os.makedirs(last_run_models_dir, exist_ok=True)
    result = train_walk_forward_regression(
        data_path=app_cfg.data_path,
        features_path=app_cfg.features_path,
        model_cfg=app_cfg.model,
        train_cfg=app_cfg.train,
        models_dir=last_run_models_dir,
        save_models=True,
        model_prefix="fold_rv",
        run_name="train_rv",
    )
    pred_df: pd.DataFrame = result["predictions_df"]
    feature_columns = list(result.get("feature_columns", []))
    target_columns = list(result.get("target_columns", []))
    metrics_per_fold = list(result.get("metrics_per_fold", []))
    pred_path = os.path.join(PREDICTIONS_DIR, "predictions_walkforward_transformer_rv.csv")
    pred_df.to_csv(pred_path, index=False)
    _emit(f"[RV] Saved predictions: {pred_path}")
    _save_rv_visualizations(pred_df)
    _emit(
        f"[RV] Predictions shape: {pred_df.shape}, "
        f"fold_ids={sorted(pred_df['fold_id'].dropna().astype(int).unique().tolist())}"
    )
    _print_feature_summary(feature_columns)
    _print_rv_metrics(metrics_per_fold, target_columns)
    _print_rv_fold_diagnostics(metrics_per_fold, target_columns)
    _print_prediction_diagnostics(pred_df, target_columns)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train-rv"], default="train-rv")
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--features-path", type=str, default=None)
    p.add_argument(
        "--model-type",
        choices=["patch_encoder", "encoder_only", "decoder_only", "vanilla_enc_dec", "patch_decoder"],
        default=None,
    )
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=None)
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="'cuda' fails if GPU is unavailable; 'cpu' forces CPU even if CUDA is present.",
    )
    return p.parse_args()


def apply_overrides(app_cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    model_dict = asdict(app_cfg.model)
    train_dict = asdict(app_cfg.train)
    for k, v in {
        "model_type": args.model_type,
        "seq_len": args.seq_len,
        "patch_size": args.patch_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
    }.items():
        if v is not None:
            model_dict[k] = v
    for k, v in {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "n_splits": args.n_splits,
    }.items():
        if v is not None:
            train_dict[k] = v

    return AppConfig(
        data_path=args.data_path or app_cfg.data_path,
        features_path=args.features_path or app_cfg.features_path,
        model=ModelConfig(**model_dict),
        train=TrainConfig(**train_dict),
    )


def main() -> None:
    _setup_logger()
    args = parse_args()
    _emit_device_diagnostics(args.device)
    device = _configure_device(args.device)
    _emit(f"[RV] effective_device    : {device}")
    app_cfg = apply_overrides(AppConfig(), args)
    _emit_json("CLI args", vars(args))
    run_train_mode(app_cfg)


if __name__ == "__main__":
    main()
