"""Historical mean baseline for RV forecasting."""

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
    return_predictions: bool = False,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)
    y = df[tgt_cols].astype(float).values
    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []
    pred_parts = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        y_train = y[train_idx]
        y_test = y[test_idx]
        mean_vec = y_train.mean(axis=0, keepdims=True)
        y_pred = np.repeat(mean_vec, repeats=len(y_test), axis=0)
        metrics_per_fold.append(
            compute_regression_metrics(y_true=y_test, y_pred=y_pred, target_columns=tgt_cols)
        )
        if return_predictions:
            part = df.iloc[test_idx][["ts"]].copy() if "ts" in df.columns else df.iloc[test_idx].copy()
            part = part.reset_index(drop=True)
            part["fold_id"] = int(fold_id)
            for i, col in enumerate(tgt_cols):
                part[f"actual_{col}"] = y_test[:, i]
                part[f"pred_{col}"] = y_pred[:, i]
            pred_parts.append(part)

    print_regression_metrics(metrics_per_fold, "Historical Mean", tgt_cols)
    if not return_predictions:
        return metrics_per_fold
    import pandas as pd
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return {"metrics_per_fold": metrics_per_fold, "predictions_df": pred_df}


if __name__ == "__main__":
    run()
