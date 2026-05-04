"""Run the full spike_warning pipeline: define → features → EDA → train → evaluate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_step(description: str, argv: list[str]) -> None:
    print(f"\n{'=' * 60}\n{description}\n{'=' * 60}")
    print("$", " ".join(argv))
    subprocess.run(argv, check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Orchestrate spike_warning: define_spikes → extract_features → "
        "analyze_pre_spike → train_classifier → evaluate",
    )
    p.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Passed to define_spikes and extract_features (0 = full history).",
    )
    p.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip analyze_pre_spike.py",
    )
    p.add_argument(
        "--preferred-model",
        default="lgb",
        choices=("lr", "rf", "lgb"),
        help="Model whose probabilities evaluate.py uses (default: lgb).",
    )
    p.add_argument(
        "--threshold-method",
        default="global_quantile",
        choices=("global_quantile", "rolling_quantile"),
        help="Passed to define_spikes.",
    )
    p.add_argument("--rv-col", default="rv_3bar", help="Passed to define_spikes.")
    p.add_argument("--quantile", type=float, default=0.95, help="Passed to define_spikes.")
    p.add_argument("--rolling-window", type=int, default=8640, help="Rolling quantile window.")
    p.add_argument("--forward-window", type=int, default=48, help="Passed to define_spikes.")
    p.add_argument("--cooldown", type=int, default=12, help="Passed to define_spikes.")
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Passed to train_classifier.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Passed to train_classifier.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    exe = sys.executable
    root = Path(__file__).resolve().parent

    limit: list[str] = []
    if args.limit_rows and args.limit_rows > 0:
        limit = ["--limit-rows", str(args.limit_rows)]

    _run_step(
        "Step 1/5: define_spikes",
        [
            exe,
            "-m",
            "spike_warning.define_spikes",
            "--rv-col",
            args.rv_col,
            "--quantile",
            str(args.quantile),
            "--threshold-method",
            args.threshold_method,
            "--rolling-window",
            str(args.rolling_window),
            "--forward-window",
            str(args.forward_window),
            "--cooldown",
            str(args.cooldown),
            *limit,
        ],
    )

    _run_step(
        "Step 2/5: extract_features",
        [exe, "-m", "spike_warning.extract_features", *limit],
    )

    if not args.skip_eda:
        _run_step(
            "Step 3/5: analyze_pre_spike",
            [exe, "-m", "spike_warning.analyze_pre_spike"],
        )
    else:
        print("\n[skip] analyze_pre_spike (--skip-eda)")

    _run_step(
        "Step 4/5: train_classifier",
        [
            exe,
            "-m",
            "spike_warning.train_classifier",
            "--test-ratio",
            str(args.test_ratio),
            "--val-ratio",
            str(args.val_ratio),
        ],
    )

    _run_step(
        "Step 5/5: evaluate",
        [
            exe,
            "-m",
            "spike_warning.evaluate",
            "--preferred-model",
            args.preferred_model,
        ],
    )

    print(f"\nDone. Artifacts under: {root / 'output'}")


if __name__ == "__main__":
    main()
