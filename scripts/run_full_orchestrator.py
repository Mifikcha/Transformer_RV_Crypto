"""
Full project orchestrator.

Sequentially runs:
1) dataset/get_data/_get_data.py
2) target/form_target.py
3) scripts/run_rv_pipeline.py — add RV targets, feature selection, train-rv, baselines, arch

Feature selection and training are part of the RV pipeline (see scripts/run_rv_pipeline.py).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _progress(step: int, total: int, message: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {message}")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return
    started = time.time()
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    print(f"[OK ] {' '.join(cmd)} (elapsed {time.time() - started:.1f}s)")


def _require_exists(rel_path: str) -> None:
    abs_path = PROJECT_ROOT / rel_path
    if not abs_path.exists():
        raise FileNotFoundError(f"Required file is missing: {abs_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full project pipeline orchestrator.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--smoke", action="store_true", help="Use light mode for heavy steps.")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset orchestrator.")
    parser.add_argument("--skip-target", action="store_true", help="Skip target orchestrator.")
    parser.add_argument("--skip-rv-pipeline", action="store_true", help="Skip scripts/run_rv_pipeline.py.")
    parser.add_argument(
        "--skip-feature-selection",
        action="store_true",
        help="Pass to RV pipeline: skip if recommended_features.csv already exists.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Pass to RV pipeline: skip baselines step.",
    )
    parser.add_argument("--skip-arch", action="store_true", help="Pass to RV pipeline: skip architecture comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_total = time.time()

    _require_exists("dataset/get_data/_get_data.py")
    _require_exists("target/form_target.py")
    _require_exists("scripts/run_rv_pipeline.py")

    steps_enabled = [
        not args.skip_dataset,
        not args.skip_target,
        not args.skip_rv_pipeline,
    ]
    total = sum(int(v) for v in steps_enabled)
    step = 0

    if total == 0:
        print("No steps selected. Nothing to run.")
        return

    if not args.skip_dataset:
        step += 1
        _progress(step, total, "Dataset orchestrator")
        _run([PY, "dataset/get_data/_get_data.py"], dry_run=args.dry_run)

    if not args.skip_target:
        step += 1
        _progress(step, total, "Target orchestrator")
        _run([PY, "target/form_target.py"], dry_run=args.dry_run)

    if not args.skip_rv_pipeline:
        step += 1
        _progress(step, total, "Scripts orchestrator (RV pipeline)")
        cmd = [PY, "scripts/run_rv_pipeline.py"]
        if args.smoke:
            cmd.append("--smoke")
        if args.skip_feature_selection:
            cmd.append("--skip-feature-selection")
        if args.skip_baselines:
            cmd.append("--skip-baselines")
        if args.skip_arch:
            cmd.append("--skip-arch")
        _run(cmd, dry_run=args.dry_run)

    print(f"[DONE] Full orchestrator finished in {time.time() - started_total:.1f}s.")


if __name__ == "__main__":
    main()
