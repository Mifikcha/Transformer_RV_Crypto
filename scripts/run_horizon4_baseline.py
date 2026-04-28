"""
Long-horizon plan, Step 1: 4-horizon baseline.

Trains the current best architecture (patch_encoder, seq_len=240, patch_size=12,
HAR full) on all 4 forward RV horizons with pure QLIKE loss (alpha=0.0, the
ablation B1.4 winner) and saves per-horizon metrics to:

    transformer/output/experiments/horizon4_baseline.csv

Columns include r2/mse/mae/qlike/bias/corr per horizon (rv_3bar_fwd,
rv_12bar_fwd, rv_48bar_fwd, rv_288bar_fwd) plus aggregate means.

Usage:
    python scripts/run_horizon4_baseline.py
    python scripts/run_horizon4_baseline.py --quick           # smoke (small folds/epochs)
    python scripts/run_horizon4_baseline.py --model-type decoder_only

The full ``train-rv`` orchestrator (run_transformer.py) is unchanged; this
script is a thin wrapper around ablation_utils.run_single_experiment to keep
the experiment-CSV format consistent with B-series ablations.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.ablation_utils import progress, run_single_experiment, save_experiment_csv
from transformer.config import AppConfig, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Long-horizon plan Step 1: 4-horizon baseline (QLIKE loss).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: fewer epochs/folds, for code validation only.",
    )
    p.add_argument(
        "--model-type",
        type=str,
        default="patch_encoder",
        help="Architecture: patch_encoder|decoder_only|patch_decoder|vanilla_enc_dec.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    ensure_output_dirs()

    model_cfg = replace(cfg.model, model_type=args.model_type)
    train_cfg = cfg.train  # already 4-horizon + alpha=0.0 by config.py defaults

    if args.quick:
        train_cfg = replace(train_cfg, max_epochs=10, n_splits=2, patience=3)

    # Sanity: ensure config defaults match Step 1 expectations.
    expected_targets = (
        "rv_3bar_fwd",
        "rv_12bar_fwd",
        "rv_48bar_fwd",
        "rv_288bar_fwd",
    )
    if tuple(train_cfg.target_columns) != expected_targets:
        raise RuntimeError(
            f"TrainConfig.target_columns={train_cfg.target_columns!r} does not "
            f"match Step 1 expectation {expected_targets!r}. Check config.py."
        )
    if float(train_cfg.loss_alpha) != 0.0:
        raise RuntimeError(
            f"TrainConfig.loss_alpha={train_cfg.loss_alpha!r} != 0.0; Step 1 "
            "requires pure QLIKE loss per ablation B1.4 finding."
        )

    progress(1, 2, "H1.0 starting (4-horizon baseline, QLIKE loss)")
    row = run_single_experiment(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_path=cfg.data_path,
        features_path=cfg.features_path,
        experiment_id="H1.0",
        variant_name="horizon4_baseline_qlike",
    )
    row["experiment_group"] = "H1"

    out_dir = os.path.join(PROJECT_ROOT, "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "horizon4_baseline.csv")
    save_experiment_csv([row], out_csv)
    progress(2, 2, f"Saved: {out_csv}")

    # Console summary: highlight per-horizon QLIKE/R2 vs LightGBM/Hist.Mean baselines.
    summary_keys = []
    for h in expected_targets:
        summary_keys.extend([f"r2_{h}", f"qlike_{h}", f"bias_{h}", f"corr_{h}"])
    summary_keys.extend(["r2_mean", "qlike_mean", "bias_mean", "elapsed_sec"])
    print("\n=== H1.0 horizon4_baseline summary ===")
    print(pd.Series({k: row.get(k) for k in summary_keys}).to_string())
    print(f"\nFull row CSV: {out_csv}")


if __name__ == "__main__":
    main()
