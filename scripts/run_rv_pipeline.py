"""
End-to-end RV pipeline runner:
1) add RV targets
2) feature selection -> feature_selection/output/recommended_features.csv
3) train transformer RV model
4) run RV baselines
5) run architecture comparison
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _run(cmd: list[str]) -> None:
    started = time.time()
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    print(f"[OK ] {' '.join(cmd)} (elapsed {time.time() - started:.1f}s)")


def _progress(step: int, total: int, message: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-targets", action="store_true")
    parser.add_argument(
        "--skip-feature-selection",
        action="store_true",
        help="Skip feature_selection (only if feature_selection/output/recommended_features.csv already exists).",
    )
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-arch", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.time()
    args = parse_args()
    total = (
        1
        + int(not args.skip_targets)
        + int(not args.skip_feature_selection)
        + 1  # train
        + int(not args.skip_baselines)
        + int(not args.skip_arch)
    )
    step = 0

    step += 1
    _progress(step, total, "Initialized RV pipeline run")
    if not args.skip_targets:
        step += 1
        _progress(step, total, "Step: generate RV targets")
        _run([PY, "scripts/add_rv_targets.py"])

    if not args.skip_feature_selection:
        step += 1
        _progress(step, total, "Step: feature selection (recommended_features.csv)")
        _run([PY, "feature_selection/run_feature_selection.py"])

    train_cmd = [PY, "transformer/run_transformer.py", "--mode", "train-rv"]
    if args.smoke:
        train_cmd += ["--max-epochs", "3", "--n-splits", "2", "--patience", "2"]
    step += 1
    _progress(step, total, "Step: train Transformer RV model")
    _run(train_cmd)

    if not args.skip_baselines:
        step += 1
        _progress(step, total, "Step: run RV baselines")
        _run([PY, "baselines/run_baselines.py"])

    if not args.skip_arch:
        arch_cmd = [PY, "scripts/run_architecture_comparison.py"]
        if args.smoke:
            arch_cmd += ["--quick"]
        step += 1
        _progress(step, total, "Step: run architecture comparison")
        _run(arch_cmd)

    print(f"[DONE] RV pipeline finished in {time.time() - started:.1f}s.")


if __name__ == "__main__":
    main()
