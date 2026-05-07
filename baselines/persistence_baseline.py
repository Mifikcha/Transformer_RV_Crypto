"""Persistence baseline for RV.

Important
---------
The targets ``rv_*_fwd`` are *forward-looking* realized volatility, i.e. they use future bars.
Therefore, using ``rv_*_fwd[t-1]`` as a feature is look-ahead leakage: at time t you cannot
know a forward-looking target from t-1 because it depends on returns up to (t-1 + horizon).

This baseline instead predicts the last *available* realized volatility computed on a
backward-looking window of the same horizon ending at t:

  y_hat_t^{(H)} = RV_{t-H+1:t}^{(H)}

computed from either GK point volatility (preferred) or close-to-close log-returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    RV_TARGET_COLS,
    compute_regression_metrics,
    get_default_data_path,
    get_regression_target_columns,
    load_dataset,
    print_regression_metrics,
    walk_forward_split,
)

_EPS = 1e-12


def _garman_klass_point(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    o = open_.astype(float).clip(lower=_EPS)
    h = high.astype(float).clip(lower=_EPS)
    l = low.astype(float).clip(lower=_EPS)
    c = close.astype(float).clip(lower=_EPS)
    ratio_hl = (h / l).clip(lower=1.0001, upper=1e6)
    ratio_co = (c / o).clip(lower=1e-6, upper=1e6)
    ln_hl = np.log(ratio_hl)
    ln_co = np.log(ratio_co)
    inner = 0.5 * ln_hl**2 - (2 * np.log(2) - 1) * ln_co**2
    return pd.Series(np.sqrt(np.maximum(0.0, inner)), index=c.index)


def _backward_rv_from_gk_point(gk_point: pd.Series, horizon: int) -> pd.Series:
    # Backward-looking RV ending at t (no forward shift).
    return np.sqrt(gk_point.pow(2).rolling(window=horizon, min_periods=horizon).sum())


def _backward_rv_from_close(close: pd.Series, horizon: int) -> pd.Series:
    # Backward-looking close-to-close RV ending at t.
    c = close.astype(float).clip(lower=_EPS)
    r = np.log(c / c.shift(1))
    return np.sqrt(r.pow(2).rolling(window=horizon, min_periods=horizon).sum())


def _compute_past_rv_features(df: pd.DataFrame, horizons: list[int]) -> dict[int, np.ndarray]:
    """Compute backward-looking RV series for each horizon; return arrays aligned to df rows."""
    if "vol_garman_klass" in df.columns:
        gk = df["vol_garman_klass"].astype(float)
    elif all(c in df.columns for c in ("open_perp", "high_perp", "low_perp", "close_perp")):
        gk = _garman_klass_point(df["open_perp"], df["high_perp"], df["low_perp"], df["close_perp"])
    else:
        gk = None

    out: dict[int, np.ndarray] = {}
    for h in horizons:
        if gk is not None:
            s = _backward_rv_from_gk_point(gk, h)
        else:
            if "close_perp" not in df.columns:
                raise ValueError(
                    "Persistence baseline requires either vol_garman_klass / OHLC perp "
                    "or close_perp to compute backward-looking RV."
                )
            s = _backward_rv_from_close(df["close_perp"], h)
        # Avoid NaNs at the beginning without leaking future information.
        s = s.ffill()
        if s.isna().any():
            first_valid = s.dropna().iloc[0] if s.notna().any() else float("nan")
            s = s.fillna(first_valid)
        out[h] = s.to_numpy(dtype=float)
    return out


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
    horizons = [int(c.split("_")[1].replace("bar", "")) for c in tgt_cols]  # rv_{H}bar_fwd
    past_rv = _compute_past_rv_features(df, horizons)
    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []
    pred_parts = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        # Predict backward-looking RV ending at t with the matching horizon.
        y_test = y[test_idx]
        y_pred = np.column_stack([past_rv[h][test_idx] for h in horizons])
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

    print_regression_metrics(metrics_per_fold, "Persistence (past-window RV)", tgt_cols)
    if not return_predictions:
        return metrics_per_fold
    import pandas as pd
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return {"metrics_per_fold": metrics_per_fold, "predictions_df": pred_df}


if __name__ == "__main__":
    run()
