"""
End-to-end RV pipeline runner:
1) add RV targets
2) feature selection -> feature_selection/output/<symbol>/recommended_features.csv
3) train transformer RV model
4) run RV baselines
5) run architecture comparison

The active trading symbol is taken from the ``--symbol`` flag or, if absent,
from the ``SYMBOL`` environment variable (default ``BTCUSDT``). All file paths
in downstream steps are derived from that symbol so the pipeline can be run
for ETHUSDT, SOLUSDT, etc. without code changes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    started = time.time()
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
    print(f"[OK ] {' '.join(cmd)} (elapsed {time.time() - started:.1f}s)")


def _progress(step: int, total: int, message: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbol",
        default=os.environ.get("SYMBOL", "BTCUSDT"),
        help="Trading symbol (e.g. BTCUSDT, ETHUSDT). Defaults to env SYMBOL or BTCUSDT.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-targets", action="store_true")
    parser.add_argument(
        "--skip-feature-selection",
        action="store_true",
        help="Skip feature_selection (only if recommended_features.csv already exists for this symbol).",
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Skip Transformer training step (useful when only baselines/arch are needed).",
    )
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-arch", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.time()
    args = parse_args()

    symbol = (args.symbol or "BTCUSDT").strip().upper()
    symbol_lower = symbol.lower()
    os.environ["SYMBOL"] = symbol

    child_env = os.environ.copy()
    child_env["SYMBOL"] = symbol

    total = (
        1
        + int(not args.skip_targets)
        + int(not args.skip_feature_selection)
        + int(not args.skip_transformer)  # train
        + int(not args.skip_baselines)
        + int(not args.skip_arch)
    )
    step = 0

    step += 1
    _progress(step, total, f"Initialized RV pipeline run for symbol={symbol}")
    if not args.skip_targets:
        step += 1
        _progress(step, total, "Step: generate RV targets")
        targets_input = os.path.join(
            PROJECT_ROOT, "target", f"{symbol_lower}_5m_final_with_targets.csv"
        )
        _run([PY, "scripts/add_rv_targets.py", "--input", targets_input], env=child_env)

    if not args.skip_feature_selection:
        step += 1
        _progress(step, total, "Step: feature selection (recommended_features.csv)")
        _run([PY, "feature_selection/run_feature_selection.py"], env=child_env)

    if not args.skip_transformer:
        train_cmd = [PY, "transformer/run_transformer.py", "--mode", "train-rv"]
        if args.smoke:
            train_cmd += ["--max-epochs", "3", "--n-splits", "2", "--patience", "2"]
        step += 1
        _progress(step, total, "Step: train Transformer RV model")
        _run(train_cmd, env=child_env)

    if not args.skip_baselines:
        step += 1
        _progress(step, total, "Step: run RV baselines")
        _run([PY, "baselines/run_baselines.py"], env=child_env)

    if not args.skip_arch:
        arch_cmd = [PY, "scripts/run_architecture_comparison.py"]
        if args.smoke:
            arch_cmd += ["--quick"]
        step += 1
        _progress(step, total, "Step: run architecture comparison")
        _run(arch_cmd, env=child_env)

    print(f"[DONE] RV pipeline finished in {time.time() - started:.1f}s.")


if __name__ == "__main__":
    main()
