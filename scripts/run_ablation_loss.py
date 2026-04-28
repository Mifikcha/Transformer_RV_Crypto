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
    p = argparse.ArgumentParser(description="B1: loss ablation.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--model-type", type=str, default="patch_encoder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    ensure_output_dirs()
    base_model = replace(cfg.model, model_type=args.model_type)
    base_train = cfg.train
    if args.quick:
        base_train = replace(base_train, max_epochs=10, n_splits=2, patience=3)

    variants = [
        ("B1.1", "rv_log_aware_alpha_0.7", replace(base_train, loss_type="rv_log_aware", loss_alpha=0.7)),
        ("B1.2", "mse", replace(base_train, loss_type="mse")),
        ("B1.3", "rv_log_aware_alpha_1.0", replace(base_train, loss_type="rv_log_aware", loss_alpha=1.0)),
        ("B1.4", "rv_log_aware_alpha_0.0", replace(base_train, loss_type="rv_log_aware", loss_alpha=0.0)),
        ("B1.5", "rv_log_aware_alpha_0.5", replace(base_train, loss_type="rv_log_aware", loss_alpha=0.5)),
        ("B1.6", "rv_log_aware_alpha_0.9", replace(base_train, loss_type="rv_log_aware", loss_alpha=0.9)),
    ]

    rows: list[dict] = []
    total = len(variants) + 1
    progress(1, total, "B1 initialized")
    for i, (exp_id, name, train_cfg) in enumerate(variants, start=2):
        progress(i, total, f"Running {exp_id} ({name})")
        row = run_single_experiment(
            model_cfg=base_model,
            train_cfg=train_cfg,
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            experiment_id=exp_id,
            variant_name=name,
        )
        row["experiment_group"] = "B1"
        rows.append(row)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ablation_loss.csv")
    rows = sorted(rows, key=lambda r: float(r.get("qlike_mean", float("inf"))))
    save_experiment_csv(rows, out_csv)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
