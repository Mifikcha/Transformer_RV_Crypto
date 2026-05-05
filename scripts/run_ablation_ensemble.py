from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.utils import load_dataset
from scripts.ablation_utils import compute_extended_metrics, save_experiment_csv
from transformer.dataset import add_rv_har_context_columns
from tg_bot.inference import RVInference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C1: ensemble size ablation without retraining.")
    p.add_argument("--data-path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "target", "btcusdt_5m_final_with_targets.csv"))
    p.add_argument("--seq-len", type=int, default=240)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def _window_iterator(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    seq_len: int,
    use_har: bool,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    feat = df[feature_columns].astype(float).to_numpy()
    tgt = df[target_columns].astype(float).to_numpy()
    har = df[[c for c in df.columns if c.endswith("_har_w") or c.endswith("_har_m")]].astype(float).to_numpy() if use_har else None
    for end in range(seq_len - 1, len(df)):
        x = feat[end - seq_len + 1 : end + 1]
        y = tgt[end]
        h = har[end] if use_har and har is not None else np.zeros(6, dtype=np.float64)
        yield x, h, y


def main() -> None:
    root = os.path.dirname(os.path.dirname(__file__))
    args = parse_args()
    model_paths = [os.path.join(root, "model", f"fold_rv_{i}.pt") for i in range(5)]
    for p in model_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    df = load_dataset(args.data_path)
    add_rv_har_context_columns(df, mode="full")
    # Last segment beyond common train windows.
    eval_df = df.tail(20_000 if not args.quick else 4_000).reset_index(drop=True)

    subsets = [
        ("C1.1", "best_fold_3", [3]),
        ("C1.2", "worst_fold_1", [1]),
        ("C1.3", "best_3_folds", [0, 2, 3]),
        ("C1.4", "all_5_folds", [0, 1, 2, 3, 4]),
    ]
    rows: list[dict] = []

    for exp_id, name, fold_ids in subsets:
        inf = RVInference([model_paths[i] for i in fold_ids])
        use_har = True
        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []
        for x, h, y in _window_iterator(
            eval_df,
            feature_columns=inf.feature_columns,
            target_columns=inf.target_columns,
            seq_len=args.seq_len,
            use_har=use_har,
        ):
            pred = inf.predict(x, h)
            y_pred_list.append(np.array([pred[col.replace("_fwd", "")] for col in inf.target_columns], dtype=float))
            y_true_list.append(y.astype(float))
        y_true = np.vstack(y_true_list)
        y_pred = np.vstack(y_pred_list)
        metrics = compute_extended_metrics(y_true=y_true, y_pred=y_pred, target_columns=inf.target_columns)
        row = {
            "experiment_group": "C1",
            "experiment_id": exp_id,
            "variant_name": name,
            "fold_ids": ",".join(str(i) for i in fold_ids),
            "n_models": len(fold_ids),
            "n_samples": int(y_true.shape[0]),
        }
        row.update(metrics)
        rows.append(row)

    out_dir = os.path.join(root, "transformer", "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ablation_ensemble.csv")
    rows = sorted(rows, key=lambda r: float(r.get("qlike_mean", float("inf"))))
    save_experiment_csv(rows, out_csv)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
