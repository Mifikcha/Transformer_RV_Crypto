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
from transformer.dataset import load_base_dataframe, load_recommended_features, resolve_features


GROUP_MAP = {
    "volatility_rv": [
        "rv_gk_15min",
        "rv_gk_240min",
        "rv_parkinson_15min",
        "rv_parkinson_240min",
        "rv_rs_240min",
        "realized_vol_240min",
        "vol_garman_klass",
        "atr_14_norm",
    ],
    "time": ["sin_day", "cos_day", "weekday", "min_to_ny_open", "min_to_asia_open"],
    "derivatives": ["fundingRate", "openInterest"],
    "price_sma": ["sma_60min", "sma_240min", "open_perp", "high_perp"],
    "volume": ["rolling_vol_std_240min"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D1: feature group ablation on transformer.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--model-type", type=str, default="patch_encoder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    ensure_output_dirs()
    base_train = cfg.train
    if args.quick:
        base_train = replace(base_train, max_epochs=10, n_splits=2, patience=3)
    else:
        base_train = replace(base_train, max_epochs=50, n_splits=3)

    df = load_base_dataframe(cfg.data_path)
    rec = load_recommended_features(cfg.features_path)
    base_features = resolve_features(df, rec)

    variants: list[tuple[str, str, list[str]]] = [("D1.0", "none", base_features)]
    for group_name, group_features in GROUP_MAP.items():
        drop_set = set(group_features)
        kept = [f for f in base_features if f not in drop_set]
        variants.append((f"D1.{len(variants)}", f"drop_{group_name}", kept))

    rows: list[dict] = []
    total = len(variants) + 1
    progress(1, total, "D1 initialized")
    for i, (exp_id, name, feature_cols) in enumerate(variants, start=2):
        progress(i, total, f"Running {exp_id} ({name})")
        row = run_single_experiment(
            model_cfg=replace(cfg.model, model_type=args.model_type),
            train_cfg=base_train,
            data_path=cfg.data_path,
            features_path=cfg.features_path,
            experiment_id=exp_id,
            variant_name=name,
            feature_cols_override=feature_cols,
        )
        row["experiment_group"] = "D1"
        rows.append(row)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ablation_features.csv")
    rows = sorted(rows, key=lambda r: float(r.get("qlike_mean", float("inf"))))
    save_experiment_csv(rows, out_csv)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
