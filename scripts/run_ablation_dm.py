"""
Diebold–Mariano + block-bootstrap CI between two saved ablation prediction streams.

Expects Parquet files from ``run_single_experiment`` (saved by default under
``<repo>/transformer/output/experiments/preds/<experiment_id>.parquet``): columns
``ts`` (optional), ``actual_<target>``, ``pred_<target>``, plus fold metadata
from training.

Example::

    python scripts/run_ablation_dm.py \\
        transformer/output/experiments/preds/B1.4.parquet \\
        transformer/output/experiments/preds/B1.2.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.ablation_utils import dm_bootstrap_compare_pred_dfs


def _infer_target_columns(df: pd.DataFrame) -> list[str]:
    """Preserve dataframe column order (matches training horizon order)."""
    cols = [c[len("pred_") :] for c in df.columns if c.startswith("pred_")]
    if not cols:
        raise ValueError("No pred_* columns found; cannot infer target_columns.")
    return cols


def _validate_pred_df(df: pd.DataFrame, target_columns: list[str], label: str) -> None:
    missing: list[str] = []
    for c in target_columns:
        for prefix in ("actual_", "pred_"):
            col = f"{prefix}{c}"
            if col not in df.columns:
                missing.append(col)
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DM test + block-bootstrap CI between two ablation pred parquet files."
    )
    p.add_argument("baseline_parquet", help="Path to baseline predictions (.parquet)")
    p.add_argument("variant_parquet", help="Path to variant predictions (.parquet)")
    p.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated target names (default: infer from pred_* columns in baseline).",
    )
    p.add_argument("--block-size", type=int, default=48)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument(
        "--hac-lags",
        type=int,
        default=-1,
        help="HAC lags for DM (default: block_size - 1). Use -1 for default.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional path to write results as JSON.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    baseline = pd.read_parquet(args.baseline_parquet)
    variant = pd.read_parquet(args.variant_parquet)

    if args.targets.strip():
        target_columns = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        target_columns = _infer_target_columns(baseline)

    _validate_pred_df(baseline, target_columns, "baseline")
    _validate_pred_df(variant, target_columns, "variant")

    hac_lags = None if args.hac_lags < 0 else int(args.hac_lags)
    if hac_lags is None and args.block_size > 0:
        hac_lags = int(args.block_size) - 1

    result = dm_bootstrap_compare_pred_dfs(
        pred_m0=baseline,
        pred_m1=variant,
        target_columns=target_columns,
        block_size=int(args.block_size),
        n_boot=int(args.n_boot),
        hac_lags=hac_lags,
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.out_json.strip():
        out_path = args.out_json.strip()
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Wrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
