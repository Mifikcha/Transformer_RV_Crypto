"""
Long-horizon plan, Step 2: horizon-specific loss weighting.

Trains the same architecture as Step 1 (patch_encoder, seq_len=240, patch_size=12,
HAR full, pure QLIKE) on all 4 forward RV horizons and sweeps 5 different
``horizon_weights`` schemes. The motivation is that with equal weights, the
short horizons (rv_3bar_fwd, rv_12bar_fwd) dominate the gradient because their
absolute QLIKE/Huber contributions are larger; the long horizons receive too
little signal to learn well.

Variants (per plan):
    H2.0 equal             0.25 / 0.25 / 0.25 / 0.25       <- regression check
    H2.1 inverse_horizon   0.10 / 0.15 / 0.25 / 0.50
    H2.2 sqrt_horizon      0.13 / 0.18 / 0.27 / 0.42
    H2.3 only_long         0.00 / 0.00 / 0.50 / 0.50
    H2.4 long_focus        0.05 / 0.10 / 0.35 / 0.50

H2.0 should be statistically indistinguishable from Step 1's horizon4_baseline.csv
because, with horizon_weights=None and the new ``reduction='none'`` path, the
loss is mathematically identical to the legacy implementation.

Output: transformer/output/experiments/horizon_weighting.csv
        sorted ascending by qlike_rv_48bar_fwd (smaller QLIKE = better).

Usage:
    python scripts/run_horizon_weighting.py
    python scripts/run_horizon_weighting.py --quick           # smoke (small folds/epochs)
    python scripts/run_horizon_weighting.py --model-type decoder_only
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


HORIZON_VARIANTS: list[tuple[str, str, tuple[float, float, float, float]]] = [
    ("H2.0", "equal",            (0.25, 0.25, 0.25, 0.25)),
    ("H2.1", "inverse_horizon",  (0.10, 0.15, 0.25, 0.50)),
    ("H2.2", "sqrt_horizon",     (0.13, 0.18, 0.27, 0.42)),
    ("H2.3", "only_long",        (0.00, 0.00, 0.50, 0.50)),
    ("H2.4", "long_focus",       (0.05, 0.10, 0.35, 0.50)),
]


def _format_weights(weights: tuple[float, ...]) -> str:
    return "/".join(f"{w:.2f}" for w in weights)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Long-horizon plan Step 2: horizon-specific loss weighting sweep.",
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
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of variant IDs to run (e.g. 'H2.1,H2.4'). "
             "Default = all 5 variants.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    ensure_output_dirs()

    base_model = replace(cfg.model, model_type=args.model_type)
    base_train = cfg.train  # 4-horizon + alpha=0.0 from Step 1

    if args.quick:
        base_train = replace(base_train, max_epochs=10, n_splits=2, patience=3)

    # Sanity: enforce Step 1 prereqs.
    expected_targets = (
        "rv_3bar_fwd",
        "rv_12bar_fwd",
        "rv_48bar_fwd",
        "rv_288bar_fwd",
    )
    if tuple(base_train.target_columns) != expected_targets:
        raise RuntimeError(
            f"TrainConfig.target_columns={base_train.target_columns!r} != "
            f"{expected_targets!r}. Step 1 must run first (4-horizon baseline)."
        )
    if float(base_train.loss_alpha) != 0.0:
        raise RuntimeError(
            f"TrainConfig.loss_alpha={base_train.loss_alpha!r} != 0.0; Step 2 "
            "uses pure QLIKE loss (Step 1 / B1.4 winner)."
        )

    selected_ids: set[str] | None = None
    if args.only:
        selected_ids = {token.strip() for token in args.only.split(",") if token.strip()}
        unknown = selected_ids - {vid for vid, _, _ in HORIZON_VARIANTS}
        if unknown:
            raise ValueError(f"Unknown variant ids in --only: {sorted(unknown)}")

    variants = [
        (vid, name, weights)
        for vid, name, weights in HORIZON_VARIANTS
        if selected_ids is None or vid in selected_ids
    ]
    if not variants:
        raise RuntimeError("No variants selected (filter --only excluded everything).")

    rows: list[dict] = []
    total = len(variants) + 1
    progress(1, total, "H2 horizon weighting sweep initialized")
    for i, (vid, name, weights) in enumerate(variants, start=2):
        train_cfg = replace(base_train, horizon_weights=weights)
        weight_scheme = _format_weights(weights)
        progress(i, total, f"Running {vid} ({name}) weights={weight_scheme}")
        row = run_single_experiment(
            model_cfg=base_model,
            train_cfg=train_cfg,
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            experiment_id=vid,
            variant_name=name,
        )
        row["experiment_group"] = "H2"
        row["weight_scheme"] = weight_scheme
        for hi, h in enumerate(expected_targets):
            row[f"weight_{h}"] = float(weights[hi])
        rows.append(row)

    # Sort by primary plan metric: QLIKE on rv_48bar_fwd (lower is better).
    rows = sorted(rows, key=lambda r: float(r.get("qlike_rv_48bar_fwd", float("inf"))))

    out_dir = os.path.join(PROJECT_ROOT, "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "horizon_weighting.csv")
    save_experiment_csv(rows, out_csv)

    # Console summary tailored to the long-horizon plan: per-horizon QLIKE +
    # mean R2/QLIKE, sorted by qlike on the 48-bar (4 hour) horizon.
    summary_cols = [
        "experiment_id",
        "variant_name",
        "weight_scheme",
        "qlike_rv_3bar_fwd",
        "qlike_rv_12bar_fwd",
        "qlike_rv_48bar_fwd",
        "qlike_rv_288bar_fwd",
        "r2_rv_3bar_fwd",
        "r2_rv_12bar_fwd",
        "r2_rv_48bar_fwd",
        "r2_rv_288bar_fwd",
        "qlike_mean",
        "r2_mean",
        "elapsed_sec",
    ]
    print("\n=== H2 horizon weighting summary (sorted by qlike_rv_48bar_fwd asc) ===")
    print(pd.DataFrame(rows)[summary_cols].to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
