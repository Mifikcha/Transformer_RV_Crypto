"""Persistence baseline for RV (predict last observed RV)."""

from __future__ import annotations

import numpy as np

from utils import (
    RV_TARGET_COLS,
    compute_regression_metrics,
    get_default_data_path,
    get_regression_target_columns,
    load_dataset,
    print_regression_metrics,
    walk_forward_split,
)


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)
    y = df[tgt_cols].astype(float).values
    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []

    for train_idx, test_idx in splits:
        y_test = y[test_idx]
        last_obs = y[train_idx][-1]
        y_pred = np.repeat(last_obs.reshape(1, -1), repeats=len(y_test), axis=0)
        metrics_per_fold.append(
            compute_regression_metrics(y_true=y_test, y_pred=y_pred, target_columns=tgt_cols)
        )

    print_regression_metrics(metrics_per_fold, "Persistence (last RV)", tgt_cols)
    return metrics_per_fold


if __name__ == "__main__":
    run()
