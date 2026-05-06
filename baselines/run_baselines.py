"""Orchestrator for RV baselines."""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure baselines dir is on path when running as script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

import numpy as np
from har_rv_baseline import run as run_har_rv
from har_rv_j_baseline import run as run_har_rv_j
from historical_mean_baseline import run as run_hist_mean
from lightgbm_baseline import run as run_lightgbm
from linear_regression_baseline import run as run_linear
from lstm_baseline import run as run_lstm
from persistence_baseline import run as run_persistence


def _mean_metric(metrics_per_fold: list[dict], key: str) -> float:
    vals = [m[key] for m in metrics_per_fold if key in m and isinstance(m[key], (int, float)) and not (isinstance(m[key], float) and np.isnan(m[key]))]
    return sum(vals) / len(vals) if vals else float("nan")


def run_regression(data_path: str | None = None, n_splits: int = 5, *, skip_lstm: bool = False) -> None:
    runners = [
        ("Persistence", run_persistence),
        ("Historical Mean", run_hist_mean),
        ("HAR-RV (Ridge)", run_har_rv),
        ("HAR-RV-J (Ridge)", run_har_rv_j),
        ("Linear Regression (Ridge)", run_linear),
        ("LightGBM", run_lightgbm),
    ]
    if not skip_lstm:
        runners.append(("LSTM (2-layer)", run_lstm))
    results = []
    for name, run_fn in runners:
        start = time.perf_counter()
        metrics_per_fold = run_fn(data_path=data_path, n_splits=n_splits)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "model": name,
                "mse_mean": _mean_metric(metrics_per_fold, "mse_mean"),
                "mae_mean": _mean_metric(metrics_per_fold, "mae_mean"),
                "r2_mean": _mean_metric(metrics_per_fold, "r2_mean"),
                "hmse_mean": _mean_metric(metrics_per_fold, "hmse_mean"),
                "qlike_mean": _mean_metric(metrics_per_fold, "qlike_mean"),
                "time_sec": round(elapsed, 2),
            }
        )

    print("\n" + "=" * 90)
    print("  BASELINE SUMMARY (REGRESSION)")
    print("=" * 90)
    print(
        f"  {'Model':<28}  {'MSE(mean)':>10}  {'MAE(mean)':>10}  "
        f"{'R2(mean)':>9}  {'HMSE':>10}  {'QLIKE':>10}  {'Time(s)':>8}"
    )
    print("-" * 90)
    for row in results:
        print(
            f"  {row['model']:<28}  {row['mse_mean']:>10.6f}  {row['mae_mean']:>10.6f}  "
            f"{row['r2_mean']:>9.4f}  {row['hmse_mean']:>10.6f}  {row['qlike_mean']:>10.6f}  {row['time_sec']:>8.2f}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--skip-lstm", action="store_true", help="Skip the slow LSTM baseline.")
    args = p.parse_args()
    run_regression(data_path=args.data_path, n_splits=int(args.n_splits), skip_lstm=bool(args.skip_lstm))
