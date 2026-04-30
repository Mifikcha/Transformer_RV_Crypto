"""
Full project orchestrator.

Sequentially runs:
1) dataset/get_data/_get_data.py
2) target/form_target.py
3) scripts/run_rv_pipeline.py — add RV targets, feature selection, train-rv, baselines

Feature selection and training are part of the RV pipeline (see
``scripts/run_rv_pipeline.py``).

Multi-symbol support
--------------------
By default the orchestrator works with ``BTCUSDT``. To run the entire
pipeline on another Bybit linear-perpetual symbol, pass either the
canonical ``--symbol XXXUSDT`` flag or any shorthand of the form
``--XXXUSDT`` (e.g. ``--ETHUSDT``, ``--SOLUSDT``). The selected symbol
is exported as the ``SYMBOL`` environment variable so every downstream
step (dataset, target, feature selection, transformer, baselines) picks
it up automatically.

Examples::

  python scripts/run_full_orchestrator.py
  python scripts/run_full_orchestrator.py --ETHUSDT
  python scripts/run_full_orchestrator.py --symbol SOLUSDT --smoke
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# Matches CLI shorthands like --ETHUSDT, --BTCUSDT, --SOLUSDT, --1000PEPEUSDT, etc.
_SYMBOL_FLAG_RE = re.compile(r"^--([A-Z0-9]+USDT)$")


def _extract_symbol_shorthand(argv: list[str]) -> tuple[list[str], str | None]:
    """Strip any ``--XXXUSDT`` shorthand from argv and return the captured symbol.

    The shorthand is mutually exclusive with ``--symbol``; if both are given,
    the shorthand wins (most specific intent) and a notice is printed.
    """
    cleaned: list[str] = []
    captured: str | None = None
    for token in argv:
        m = _SYMBOL_FLAG_RE.match(token)
        if m:
            sym = m.group(1)
            if captured is not None and captured != sym:
                print(
                    f"[WARN] Multiple --<SYMBOL>USDT shorthands provided "
                    f"({captured} and {sym}); using the last one: {sym}.",
                    file=sys.stderr,
                )
            captured = sym
            continue
        cleaned.append(token)
    return cleaned, captured


def _progress(step: int, total: int, message: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {message}")


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return
    started = time.time()
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)
    print(f"[OK ] {' '.join(cmd)} (elapsed {time.time() - started:.1f}s)")


def _require_exists(rel_path: str) -> None:
    abs_path = PROJECT_ROOT / rel_path
    if not abs_path.exists():
        raise FileNotFoundError(f"Required file is missing: {abs_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full project pipeline orchestrator. "
            "Use --ETHUSDT (or any --XXXUSDT shorthand) / --symbol XXXUSDT "
            "to switch the active trading symbol; defaults to BTCUSDT."
        )
    )
    parser.add_argument(
        "--symbol",
        default=os.environ.get("SYMBOL", "BTCUSDT"),
        help="Bybit linear-perpetual symbol (e.g. BTCUSDT, ETHUSDT, SOLUSDT).",
    )
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
    return parser.parse_args(argv)


def main() -> None:
    raw_argv = sys.argv[1:]
    cleaned_argv, shorthand_symbol = _extract_symbol_shorthand(raw_argv)
    args = parse_args(cleaned_argv)

    # Shorthand overrides --symbol when both are provided.
    symbol = (shorthand_symbol or args.symbol or "BTCUSDT").strip().upper()
    if not symbol.endswith("USDT"):
        raise SystemExit(
            f"--symbol expects a Bybit USDT-quoted symbol (got {symbol!r}). "
            "Examples: BTCUSDT, ETHUSDT, SOLUSDT."
        )

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

    print(f"[INFO] Active symbol: {symbol}")

    # Single source of truth for downstream subprocesses: SYMBOL env var.
    os.environ["SYMBOL"] = symbol
    child_env = os.environ.copy()
    child_env["SYMBOL"] = symbol

    if not args.skip_dataset:
        step += 1
        _progress(step, total, f"Dataset orchestrator (symbol={symbol})")
        _run(
            [PY, "dataset/get_data/_get_data.py", "--symbol", symbol],
            dry_run=args.dry_run,
            env=child_env,
        )

    if not args.skip_target:
        step += 1
        _progress(step, total, f"Target orchestrator (symbol={symbol})")
        _run([PY, "target/form_target.py"], dry_run=args.dry_run, env=child_env)

    if not args.skip_rv_pipeline:
        step += 1
        _progress(step, total, f"Scripts orchestrator (RV pipeline, symbol={symbol})")
        cmd = [PY, "scripts/run_rv_pipeline.py", "--symbol", symbol]
        if args.smoke:
            cmd.append("--smoke")
        if args.skip_feature_selection:
            cmd.append("--skip-feature-selection")
        if args.skip_baselines:
            cmd.append("--skip-baselines")
        # Always skip architecture comparison in full orchestrator.
        cmd.append("--skip-arch")
        _run(cmd, dry_run=args.dry_run, env=child_env)

    print(f"[DONE] Full orchestrator finished in {time.time() - started_total:.1f}s.")


if __name__ == "__main__":
    main()
