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
    p = argparse.ArgumentParser(description="A4: patch size ablation.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--model-type", type=str, default="patch_encoder", choices=["patch_encoder", "patch_decoder"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    ensure_output_dirs()
    train_cfg = cfg.train
    if args.quick:
        train_cfg = replace(train_cfg, max_epochs=10, n_splits=2, patience=3)

    variants = [
        ("A4.1", "patch_1", 1),
        ("A4.2", "patch_6", 6),
        ("A4.3", "patch_12", 12),
        ("A4.4", "patch_24", 24),
        ("A4.5", "patch_48", 48),
    ]

    rows: list[dict] = []
    total = len(variants) + 1
    progress(1, total, f"A4 initialized | model_type={args.model_type}")
    for i, (exp_id, name, patch_size) in enumerate(variants, start=2):
        model_cfg = replace(cfg.model, model_type=args.model_type, seq_len=240, patch_size=patch_size)
        progress(i, total, f"Running {exp_id} ({name})")
        row = run_single_experiment(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            experiment_id=exp_id,
            variant_name=name,
        )
        row["experiment_group"] = "A4"
        rows.append(row)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ablation_patch_size.csv")
    rows = sorted(rows, key=lambda r: float(r.get("qlike_mean", float("inf"))))
    save_experiment_csv(rows, out_csv)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
