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
    p = argparse.ArgumentParser(description="A2: HAR context ablation.")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    ensure_output_dirs()
    if args.quick:
        train_cfg = replace(cfg.train, max_epochs=10, n_splits=2, patience=3)
    else:
        train_cfg = cfg.train

    variants = [
        ("A2.1", "har_full", replace(cfg.model, model_type="patch_encoder", har_mode="full")),
        ("A2.2", "har_none", replace(cfg.model, model_type="patch_encoder", har_mode="none")),
        ("A2.3", "har_weekly_only", replace(cfg.model, model_type="patch_encoder", har_mode="weekly_only")),
        ("A2.4", "decoder_only_ref", replace(cfg.model, model_type="decoder_only", har_mode="none")),
    ]

    rows: list[dict] = []
    total = len(variants) + 1
    progress(1, total, "A2 initialized")
    for i, (exp_id, name, model_cfg) in enumerate(variants, start=2):
        progress(i, total, f"Running {exp_id} ({name})")
        row = run_single_experiment(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            experiment_id=exp_id,
            variant_name=name,
        )
        row["experiment_group"] = "A2"
        rows.append(row)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ablation_har.csv")
    rows = sorted(rows, key=lambda r: float(r.get("qlike_mean", float("inf"))))
    save_experiment_csv(rows, out_csv)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
