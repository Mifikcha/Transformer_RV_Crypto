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
    p = argparse.ArgumentParser(description="A3: seq_len ablation.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--model-type", type=str, default="patch_encoder")
    return p.parse_args()


def _best_architecture(default_model: str, out_dir: str) -> str:
    path = os.path.join(out_dir, "architecture_comparison.csv")
    if not os.path.exists(path):
        return default_model
    df = pd.read_csv(path)
    if "model_type" not in df.columns:
        return default_model
    df = df[~df["model_type"].astype(str).str.startswith("baseline:")].copy()
    if df.empty:
        return default_model
    if "qlike_mean" in df.columns:
        df = df.sort_values("qlike_mean", ascending=True)
    elif "mse_mean" in df.columns:
        df = df.sort_values("mse_mean", ascending=True)
    return str(df.iloc[0]["model_type"])


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    ensure_output_dirs()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)

    model_type = _best_architecture(args.model_type, out_dir)
    base_train = cfg.train
    if args.quick:
        base_train = replace(base_train, max_epochs=10, n_splits=2, patience=3)

    variants = [
        ("A3.1", "seq_48", 48, 12, None),
        ("A3.2", "seq_120", 120, 12, None),
        ("A3.3", "seq_240", 240, 12, None),
        ("A3.4", "seq_480", 480, 12, 64),
        ("A3.5", "seq_576", 576, 12, 64),
    ]

    rows: list[dict] = []
    total = len(variants) + 1
    progress(1, total, f"A3 initialized | model_type={model_type}")
    for i, (exp_id, name, seq_len, patch_size, batch_size_override) in enumerate(variants, start=2):
        model_cfg = replace(cfg.model, model_type=model_type, seq_len=seq_len, patch_size=patch_size)
        train_cfg = replace(base_train, batch_size=batch_size_override) if batch_size_override else base_train
        progress(i, total, f"Running {exp_id} ({name})")
        row = run_single_experiment(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            experiment_id=exp_id,
            variant_name=name,
        )
        row["experiment_group"] = "A3"
        if batch_size_override:
            row["comment"] = "batch_size=64 for long context"
        rows.append(row)

    out_csv = os.path.join(out_dir, "ablation_seq_len.csv")
    rows = sorted(rows, key=lambda r: float(r.get("qlike_mean", float("inf"))))
    save_experiment_csv(rows, out_csv)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
