"""
Long-horizon plan, Step 3: seq_len ablation on 4 horizons.

Hypothesis: longer context may matter for distant horizons (especially rv_288bar_fwd),
even if it was not important for short horizons in the earlier 2-horizon ablation.

This script runs the 5 variants from the plan (all with patch_size=12):
    H3.1 seq_len=240   batch=128
    H3.2 seq_len=480   batch=64
    H3.3 seq_len=576   batch=64
    H3.4 seq_len=864   batch=32
    H3.5 seq_len=1152  batch=32

Loss: pure QLIKE (alpha=0.0) as per ablation B1.4 / plan Step 1.
Horizon weights: by default uses TrainConfig.horizon_weights (can be set from Step 2),
or can be overridden via CLI.

Output: transformer/output/experiments/horizon_seq_len.csv
        sorted ascending by qlike_rv_288bar_fwd (lower is better).

Usage:
    python scripts/run_horizon_seq_len.py
    python scripts/run_horizon_seq_len.py --quick
    python scripts/run_horizon_seq_len.py --model-type decoder_only
    python scripts/run_horizon_seq_len.py --horizon-weights 0.05,0.10,0.35,0.50
    python scripts/run_horizon_seq_len.py --only H3.3,H3.5
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


SEQ_LEN_VARIANTS: list[tuple[str, str, int, int]] = [
    ("H3.1", "seq_240", 240, 128),
    ("H3.2", "seq_480", 480, 64),
    ("H3.3", "seq_576", 576, 64),
    ("H3.4", "seq_864", 864, 32),
    ("H3.5", "seq_1152", 1152, 32),
]


def _parse_weights_arg(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    vals: list[float] = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    if not vals:
        return None
    return tuple(vals)


def _format_weights(weights: tuple[float, ...]) -> str:
    return "/".join(f"{w:.2f}" for w in weights)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Long-horizon plan Step 3: seq_len ablation on 4 horizons.",
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
        "--horizon-weights",
        type=str,
        default=None,
        help="Override horizon_weights as comma-separated floats (e.g. 0.05,0.10,0.35,0.50). "
        "If omitted, uses TrainConfig.horizon_weights (default: equal).",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of variant IDs to run (e.g. 'H3.2,H3.5'). "
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
            f"TrainConfig.loss_alpha={base_train.loss_alpha!r} != 0.0; Step 3 "
            "uses pure QLIKE loss (Step 1 / B1.4 winner)."
        )

    override_w = _parse_weights_arg(args.horizon_weights)
    if override_w is not None:
        base_train = replace(base_train, horizon_weights=override_w)

    selected_ids: set[str] | None = None
    if args.only:
        selected_ids = {token.strip() for token in args.only.split(",") if token.strip()}
        unknown = selected_ids - {vid for vid, _, _, _ in SEQ_LEN_VARIANTS}
        if unknown:
            raise ValueError(f"Unknown variant ids in --only: {sorted(unknown)}")

    variants = [
        (vid, name, seq_len, batch_size)
        for vid, name, seq_len, batch_size in SEQ_LEN_VARIANTS
        if selected_ids is None or vid in selected_ids
    ]
    if not variants:
        raise RuntimeError("No variants selected (filter --only excluded everything).")

    patch_size = int(base_model.patch_size)
    bad = [seq for _, _, seq, _ in variants if seq % patch_size != 0]
    if bad:
        raise ValueError(
            f"seq_len values {bad} are not divisible by patch_size={patch_size}."
        )

    hw_cfg = tuple(getattr(base_train, "horizon_weights", ()) or ())
    hw_str = _format_weights(hw_cfg) if hw_cfg else "equal"

    rows: list[dict] = []
    total = len(variants) + 1
    progress(1, total, f"H3 seq_len sweep initialized | model={base_model.model_type} hw={hw_str}")
    for i, (vid, name, seq_len, batch_size) in enumerate(variants, start=2):
        model_cfg = replace(base_model, seq_len=int(seq_len))
        train_cfg = replace(base_train, batch_size=int(batch_size))
        progress(i, total, f"Running {vid} ({name}) seq_len={seq_len} batch={batch_size}")
        row = run_single_experiment(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            experiment_id=vid,
            variant_name=name,
        )
        row["experiment_group"] = "H3"
        row["seq_len"] = int(seq_len)
        row["patch_size"] = int(patch_size)
        row["batch_size"] = int(batch_size)
        row["weight_scheme"] = hw_str
        rows.append(row)

    # Sort by primary plan metric: QLIKE on rv_288bar_fwd (lower is better).
    rows = sorted(rows, key=lambda r: float(r.get("qlike_rv_288bar_fwd", float("inf"))))

    out_dir = os.path.join(PROJECT_ROOT, "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "horizon_seq_len.csv")
    save_experiment_csv(rows, out_csv)

    summary_cols = [
        "experiment_id",
        "variant_name",
        "seq_len",
        "batch_size",
        "weight_scheme",
        "qlike_rv_48bar_fwd",
        "qlike_rv_288bar_fwd",
        "r2_rv_48bar_fwd",
        "r2_rv_288bar_fwd",
        "qlike_mean",
        "r2_mean",
        "elapsed_sec",
    ]
    print("\n=== H3 seq_len summary (sorted by qlike_rv_288bar_fwd asc) ===")
    print(pd.DataFrame(rows)[summary_cols].to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

