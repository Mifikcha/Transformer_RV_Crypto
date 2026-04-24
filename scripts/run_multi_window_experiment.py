"""
Run multi-window experiment (12 / 48 / 240 bars) for the best architecture.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformer.config import AppConfig, ensure_output_dirs
from transformer.train import train_walk_forward_regression


def _aggregate(metrics_per_fold: list[dict], key: str) -> float:
    vals = [float(m[key]) for m in metrics_per_fold if key in m]
    return float(np.mean(vals)) if vals else float("nan")


def _progress(step: int, total: int, message: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {message}")


def main() -> None:
    started = time.time()
    cfg = AppConfig()
    ensure_output_dirs()
    root_dir = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(root_dir, "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    arch_path = os.path.join(out_dir, "architecture_comparison.csv")
    _progress(1, 5, "Initialized config and output paths")
    if not os.path.exists(arch_path):
        raise FileNotFoundError(
            f"Missing {arch_path}. Run scripts/run_architecture_comparison.py first."
        )
    arch_df = pd.read_csv(arch_path).sort_values("mse_mean", ascending=True)
    best_model_type = str(arch_df.iloc[0]["model_type"])
    _progress(2, 5, f"Loaded best architecture: {best_model_type}")

    rows: list[dict] = []
    windows = (12, 48, 240)
    for i, seq_len in enumerate(windows, start=1):
        _progress(2 + i, 5, f"Training window={seq_len}")
        patch_size = cfg.model.patch_size
        if best_model_type in {"patch_encoder", "encoder_only", "patch_decoder"} and seq_len % patch_size != 0:
            patch_size = 1
        model_cfg = replace(
            cfg.model,
            model_type=best_model_type,
            seq_len=seq_len,
            patch_size=patch_size,
            n_horizons=len(cfg.train.target_columns),
        )
        result = train_walk_forward_regression(
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            model_cfg=model_cfg,
            train_cfg=cfg.train,
            save_models=False,
            model_prefix=f"{best_model_type}_{seq_len}_fold",
            run_name=f"window_{seq_len}",
        )
        rows.append(
            {
                "model_type": best_model_type,
                "seq_len": seq_len,
                "mse_mean": _aggregate(result["metrics_per_fold"], "mse_mean"),
                "mae_mean": _aggregate(result["metrics_per_fold"], "mae_mean"),
                "r2_mean": _aggregate(result["metrics_per_fold"], "r2_mean"),
                "da_mean": _aggregate(result["metrics_per_fold"], "da_mean"),
                "qlike_mean": _aggregate(result["metrics_per_fold"], "qlike_mean"),
            }
        )

    out = pd.DataFrame(rows).sort_values("seq_len")
    out_csv = os.path.join(out_dir, "multi_window_experiment.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved multi-window results: {out_csv}")
    print(out.to_string(index=False))
    _progress(5, 5, f"Finished in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
