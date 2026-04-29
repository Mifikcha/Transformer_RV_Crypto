"""
Long-horizon plan, Step 4: long-range features sweep.

Trains the same architecture as Step 1 (patch_encoder, seq_len=240, patch_size=12,
HAR full, pure QLIKE) on all 4 forward RV horizons and sweeps 6 different
feature sets that progressively add long-range signals (daily / weekly / funding
/ open-interest / regime indicator).

Variants (per plan):
    H4.0 recommended_baseline   21 features (current)
    H4.1 plus_daily_rv          23 = baseline + rv_gk_1440min, rv_parkinson_1440min
    H4.2 plus_daily_weekly_rv   24 = H4.1 + rv_gk_weekly
    H4.3 plus_daily_funding_oi  25 = H4.1 + funding_rate_std_24h, oi_change_rate_24h
    H4.4 plus_all_long_range    27 = baseline + all 6 long-range features
    H4.5 plus_daily_ratio       24 = H4.1 + rv_ratio_15min_to_daily

Pre-requisites:
  1. Step 1 config defaults (4 horizons, alpha=0.0) -- already in TrainConfig.
  2. The dataset must contain the 6 long-range columns. If they are absent
     this script will fail fast and print the command to add them:
         python scripts/add_long_range_features.py

Output: transformer/output/experiments/horizon_features.csv
        sorted ascending by qlike_rv_288bar_fwd (lower is better).

Usage:
    python scripts/run_horizon_features.py
    python scripts/run_horizon_features.py --quick
    python scripts/run_horizon_features.py --only H4.1,H4.4
    python scripts/run_horizon_features.py --horizon-weights 0.05,0.10,0.35,0.50
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
from scripts.add_long_range_features import LONG_RANGE_COLS
from transformer.config import AppConfig, ensure_output_dirs
from transformer.dataset import load_base_dataframe, load_recommended_features, resolve_features


_DAILY = ["rv_gk_1440min", "rv_parkinson_1440min"]
_WEEKLY = ["rv_gk_weekly"]
_FUNDING_OI = ["funding_rate_std_24h", "oi_change_rate_24h"]
_RATIO = ["rv_ratio_15min_to_daily"]


def _make_variants() -> list[tuple[str, str, list[str]]]:
    """Return list of (variant_id, variant_name, extra_features_to_append)."""
    return [
        ("H4.0", "recommended_baseline",  []),
        ("H4.1", "plus_daily_rv",         list(_DAILY)),
        ("H4.2", "plus_daily_weekly_rv",  list(_DAILY) + list(_WEEKLY)),
        ("H4.3", "plus_daily_funding_oi", list(_DAILY) + list(_FUNDING_OI)),
        ("H4.4", "plus_all_long_range",   list(LONG_RANGE_COLS)),
        ("H4.5", "plus_daily_ratio",      list(_DAILY) + list(_RATIO)),
    ]


def _parse_weights_arg(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    vals = [float(p) for p in parts if p]
    return tuple(vals) if vals else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Long-horizon plan Step 4: long-range features sweep.",
    )
    p.add_argument("--quick", action="store_true",
                   help="Smoke run: fewer epochs/folds, for code validation only.")
    p.add_argument("--model-type", type=str, default="patch_encoder",
                   help="Architecture: patch_encoder|decoder_only|patch_decoder|vanilla_enc_dec.")
    p.add_argument("--horizon-weights", type=str, default=None,
                   help="Override horizon_weights as comma-separated floats "
                        "(e.g. 0.05,0.10,0.35,0.50). If omitted, uses TrainConfig.horizon_weights.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated variant IDs (e.g. 'H4.1,H4.4'). Default = all 6 variants.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    ensure_output_dirs()

    base_model = replace(cfg.model, model_type=args.model_type)
    base_train = cfg.train  # 4-horizon + alpha=0.0 from Step 1

    if args.quick:
        base_train = replace(base_train, max_epochs=10, n_splits=2, patience=3)

    expected_targets = (
        "rv_3bar_fwd", "rv_12bar_fwd", "rv_48bar_fwd", "rv_288bar_fwd",
    )
    if tuple(base_train.target_columns) != expected_targets:
        raise RuntimeError(
            f"TrainConfig.target_columns={base_train.target_columns!r} != "
            f"{expected_targets!r}. Step 1 must be in effect."
        )
    if float(base_train.loss_alpha) != 0.0:
        raise RuntimeError(
            f"TrainConfig.loss_alpha={base_train.loss_alpha!r} != 0.0; Step 4 "
            "uses pure QLIKE loss (Step 1 / B1.4 winner)."
        )

    override_w = _parse_weights_arg(args.horizon_weights)
    if override_w is not None:
        base_train = replace(base_train, horizon_weights=override_w)

    df = load_base_dataframe(cfg.data_path)

    missing = [c for c in LONG_RANGE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Dataset is missing long-range columns: {missing}. "
            "Run `python scripts/add_long_range_features.py` first to add them."
        )

    rec_features = load_recommended_features(cfg.features_path)
    base_feature_cols = resolve_features(df, rec_features)
    base_n = len(base_feature_cols)
    print(f"[H4] Base recommended features: {base_n} (from {cfg.features_path})")

    variants = _make_variants()
    selected_ids: set[str] | None = None
    if args.only:
        selected_ids = {token.strip() for token in args.only.split(",") if token.strip()}
        unknown = selected_ids - {vid for vid, _, _ in variants}
        if unknown:
            raise ValueError(f"Unknown variant ids in --only: {sorted(unknown)}")
    if selected_ids is not None:
        variants = [v for v in variants if v[0] in selected_ids]
    if not variants:
        raise RuntimeError("No variants selected (filter --only excluded everything).")

    rows: list[dict] = []
    total = len(variants) + 1
    progress(1, total, f"H4 long-range features sweep initialized | base_features={base_n}")

    for i, (vid, name, extras) in enumerate(variants, start=2):
        feature_cols = list(base_feature_cols)
        for col in extras:
            if col in feature_cols:
                continue  # avoid duplicates if user already includes it in baseline
            if col not in df.columns:
                raise RuntimeError(f"{vid}: column '{col}' missing from dataset.")
            feature_cols.append(col)
        n_feat = len(feature_cols)
        extras_str = ",".join(extras) if extras else "(none)"
        progress(i, total, f"Running {vid} ({name}) n_features={n_feat} extras={extras_str}")

        row = run_single_experiment(
            model_cfg=base_model,
            train_cfg=base_train,
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            experiment_id=vid,
            variant_name=name,
            feature_cols_override=feature_cols,
        )
        row["experiment_group"] = "H4"
        row["n_features"] = n_feat
        row["extras"] = extras_str
        for col in LONG_RANGE_COLS:
            row[f"uses_{col}"] = int(col in extras)
        rows.append(row)

    # Sort by primary plan metric: QLIKE on rv_288bar_fwd (lower is better).
    rows = sorted(rows, key=lambda r: float(r.get("qlike_rv_288bar_fwd", float("inf"))))

    out_dir = os.path.join(PROJECT_ROOT, "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "horizon_features.csv")
    save_experiment_csv(rows, out_csv)

    summary_cols = [
        "experiment_id", "variant_name", "n_features", "extras",
        "qlike_rv_48bar_fwd", "qlike_rv_288bar_fwd",
        "r2_rv_48bar_fwd", "r2_rv_288bar_fwd",
        "qlike_mean", "r2_mean", "elapsed_sec",
    ]
    print("\n=== H4 long-range features summary (sorted by qlike_rv_288bar_fwd asc) ===")
    print(pd.DataFrame(rows)[summary_cols].to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
