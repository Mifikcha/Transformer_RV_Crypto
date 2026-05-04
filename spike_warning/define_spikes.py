from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from spike_warning.common import load_base_frame, save_csv


def _forward_max(series: pd.Series, window: int) -> pd.Series:
    """max(series[t+1:t+window]) for each t."""
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        j1 = i + 1
        j2 = min(i + 1 + window, n)
        if j1 >= j2:
            continue
        out[i] = np.nanmax(vals[j1:j2])
    return pd.Series(out, index=series.index)


def _time_to_first_spike(
    rv: pd.Series,
    threshold: pd.Series,
    forward_window: int,
) -> pd.Series:
    rv_vals = rv.to_numpy(dtype=float)
    thr_vals = threshold.to_numpy(dtype=float)
    n = len(rv_vals)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(thr_vals[i]):
            continue
        j1 = i + 1
        j2 = min(i + 1 + forward_window, n)
        if j1 >= j2:
            continue
        window_vals = rv_vals[j1:j2]
        mask = window_vals > thr_vals[i]
        if np.any(mask):
            first = int(np.argmax(mask))
            out[i] = float(first + 1)
    return pd.Series(out, index=rv.index)


def _apply_cooldown(signal: pd.Series, cooldown: int) -> pd.Series:
    event = np.zeros(len(signal), dtype=int)
    last = -cooldown - 1
    s = signal.fillna(0).astype(int).to_numpy()
    for i in range(len(s)):
        if s[i] == 1 and (i - last) > cooldown:
            event[i] = 1
            last = i
    return pd.Series(event, index=signal.index)


def _build_threshold(
    df: pd.DataFrame,
    rv_col: str,
    method: str,
    quantile: float,
    rolling_window: int,
) -> pd.Series:
    """Build the spike threshold series.

    The threshold defines what *counts* as a spike (it's used to label the
    target). It is not a feature seen by the model at inference time, so it is
    fair to use the full series here. A constant global quantile keeps the
    positive rate stable across train/val/test splits and avoids degenerate
    splits in non-stationary data.
    """
    if method == "global_quantile":
        thr = float(df[rv_col].quantile(quantile))
        return pd.Series(np.full(len(df), thr, dtype=float), index=df.index)
    if method == "rolling_quantile":
        return df[rv_col].rolling(rolling_window, min_periods=2880).quantile(quantile)
    raise ValueError(f"Unknown threshold method: {method}")


def label_spikes(
    df: pd.DataFrame,
    rv_col: str = "rv_3bar",
    quantile: float = 0.95,
    rolling_window: int = 8640,
    forward_window: int = 48,
    cooldown: int = 12,
    threshold_method: str = "global_quantile",
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("ts").reset_index(drop=True)

    out["spike_threshold"] = _build_threshold(
        out, rv_col=rv_col, method=threshold_method,
        quantile=quantile, rolling_window=rolling_window,
    )
    out["max_rv_next_4h"] = _forward_max(out[rv_col], forward_window)
    out["spike_in_next_4h"] = (
        out["max_rv_next_4h"] > out["spike_threshold"]
    ).fillna(False).astype(int)
    out["time_to_spike"] = _time_to_first_spike(
        out[rv_col], out["spike_threshold"], forward_window
    )
    out["spike_event"] = _apply_cooldown(out["spike_in_next_4h"], cooldown)
    return out


def _print_monthly_distribution(df: pd.DataFrame) -> None:
    if "ts" not in df.columns:
        return
    ts = pd.to_datetime(df["ts"], utc=True)
    period = ts.dt.to_period("M")
    grouped = df.assign(_period=period).groupby("_period")
    rate = grouped["spike_in_next_4h"].mean()
    counts = grouped["spike_in_next_4h"].sum().astype(int)
    totals = grouped["spike_in_next_4h"].count().astype(int)
    print("\nMonthly positive rate (label sanity check):")
    for period_idx in rate.index:
        print(
            f"  {period_idx} | rate={rate.loc[period_idx]:.2%} "
            f"| pos={counts.loc[period_idx]:>6,} | rows={totals.loc[period_idx]:>6,}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Define spike labels from historical RV")
    p.add_argument("--rv-col", default="rv_3bar")
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument(
        "--threshold-method",
        choices=("global_quantile", "rolling_quantile"),
        default="global_quantile",
        help="global_quantile (default) keeps positive rate stable across time; "
             "rolling_quantile is the original adaptive variant.",
    )
    p.add_argument("--rolling-window", type=int, default=8640)
    p.add_argument("--forward-window", type=int, default=48)
    p.add_argument("--cooldown", type=int, default=12)
    p.add_argument("--limit-rows", type=int, default=0, help="0 = full history")
    return p


def main() -> None:
    args = build_parser().parse_args()
    df = load_base_frame(limit_rows=args.limit_rows or None)
    if df.empty:
        raise RuntimeError("No data loaded from DB.")
    if args.rv_col not in df.columns:
        raise RuntimeError(f"RV column not found: {args.rv_col}")

    labels = label_spikes(
        df=df,
        rv_col=args.rv_col,
        quantile=args.quantile,
        rolling_window=args.rolling_window,
        forward_window=args.forward_window,
        cooldown=args.cooldown,
        threshold_method=args.threshold_method,
    )
    keep_cols = [
        "ts",
        args.rv_col,
        "spike_threshold",
        "max_rv_next_4h",
        "spike_in_next_4h",
        "time_to_spike",
        "spike_event",
    ]
    out = labels[keep_cols].copy()
    path = save_csv(out, "spike_labels.csv")

    n_total = len(out)
    n_pos = int(out["spike_in_next_4h"].sum())
    n_events = int(out["spike_event"].sum())
    print(f"Saved: {path}")
    print(f"Threshold method: {args.threshold_method}")
    if args.threshold_method == "global_quantile":
        print(f"Threshold value: {float(out['spike_threshold'].iloc[0]):.6f}")
    print(f"Total: {n_total:,}")
    print(f"Spikes: {n_pos:,} ({(n_pos / max(n_total, 1)):.2%})")
    print(f"Unique spike events (cooldown): {n_events:,}")
    _print_monthly_distribution(out)


if __name__ == "__main__":
    main()
