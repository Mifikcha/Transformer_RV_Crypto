"""
Long-horizon plan, Step 4: add long-range features to the dataset.

Adds 6 causal long-range features (past-only rolling windows) for the
forward RV forecasting transformer:

    rv_gk_1440min          Garman-Klass RV over 1 day  (288 bars * 5 min)
    rv_parkinson_1440min   Parkinson    RV over 1 day  (288 bars)
    rv_gk_weekly           Garman-Klass RV over 1 week (2016 bars)
    funding_rate_std_24h   std(fundingRate) over 24h
    oi_change_rate_24h     openInterest_t / openInterest_{t-288} - 1
    rv_ratio_15min_to_daily rv_gk_15min / rv_gk_1440min  (regime indicator)

All windows are past-only, no leakage. Warmup NaNs are bfill/ffill-ed (this
matches the existing convention in ``view/feature_engine.py`` for OHLC RVs).
The first ~2016 rows of the resulting dataset will have backfilled values for
``rv_gk_weekly`` -- that is < 0.5% of the full BTCUSDT 5-min history and is
acceptable for training; live inference handles the same warmup naturally.

Usage:
    python scripts/add_long_range_features.py
    python scripts/add_long_range_features.py --input PATH --output PATH
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd


DAY_BARS = 288         # 1 day  = 288 * 5min
WEEK_BARS = 7 * 288    # 1 week = 2016 bars

LONG_RANGE_COLS: list[str] = [
    "rv_gk_1440min",
    "rv_parkinson_1440min",
    "rv_gk_weekly",
    "funding_rate_std_24h",
    "oi_change_rate_24h",
    "rv_ratio_15min_to_daily",
]


DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "target",
    "btcusdt_5m_final_with_targets.csv",
)


def _gk_point(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Garman-Klass point volatility per bar (matches add_rv_targets._garman_klass_point)."""
    o = open_.clip(lower=1e-12)
    h = high.clip(lower=1e-12)
    l = low.clip(lower=1e-12)
    c = close.clip(lower=1e-12)
    ratio_hl = (h / l).clip(lower=1.0001, upper=1e6)
    ratio_co = (c / o).clip(lower=1e-6, upper=1e6)
    ln_hl = np.log(ratio_hl)
    ln_co = np.log(ratio_co)
    inner = 0.5 * ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2
    return pd.Series(np.sqrt(np.maximum(0.0, inner)), index=c.index)


def _parkinson_point(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson point volatility per bar."""
    h = high.clip(lower=1e-12)
    l = low.clip(lower=1e-12)
    ratio_hl = (h / l).clip(lower=1.0001, upper=1e6)
    ln_hl = np.log(ratio_hl)
    return pd.Series(np.sqrt(np.maximum(0.0, ln_hl ** 2 / (4.0 * np.log(2)))), index=h.index)


def add_long_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 6 long-range features (LONG_RANGE_COLS) to ``df``.

    Idempotent: if a column already exists it is overwritten.
    """
    out = df.copy()
    for col in ("close_perp",):
        if col not in out.columns:
            raise ValueError(
                f"Column '{col}' is required; did you run dataset/get_data + add_rv_targets first?"
            )

    # GK and Parkinson point series (5-minute resolution).
    if "vol_garman_klass" in out.columns:
        gk_point = out["vol_garman_klass"].astype(float)
    else:
        for col in ("open_perp", "high_perp", "low_perp"):
            if col not in out.columns:
                raise ValueError(
                    f"Need '{col}' to compute GK point volatility on the fly."
                )
        gk_point = _gk_point(
            out["open_perp"].astype(float),
            out["high_perp"].astype(float),
            out["low_perp"].astype(float),
            out["close_perp"].astype(float),
        )

    if "vol_parkinson" in out.columns:
        parkinson_point = out["vol_parkinson"].astype(float)
    else:
        for col in ("high_perp", "low_perp"):
            if col not in out.columns:
                raise ValueError(
                    f"Need '{col}' to compute Parkinson point volatility on the fly."
                )
        parkinson_point = _parkinson_point(
            out["high_perp"].astype(float), out["low_perp"].astype(float),
        )

    # 1) Daily Garman-Klass RV (288-bar rolling sum-of-squares)
    out["rv_gk_1440min"] = np.sqrt(
        gk_point.pow(2).rolling(window=DAY_BARS, min_periods=DAY_BARS).sum()
    )

    # 2) Daily Parkinson RV (288-bar)
    out["rv_parkinson_1440min"] = np.sqrt(
        parkinson_point.pow(2).rolling(window=DAY_BARS, min_periods=DAY_BARS).sum()
    )

    # 3) Weekly Garman-Klass RV (2016-bar)
    out["rv_gk_weekly"] = np.sqrt(
        gk_point.pow(2).rolling(window=WEEK_BARS, min_periods=WEEK_BARS).sum()
    )

    # 4) Funding rate std over 24h (288 bars).  In the dataset funding rate is
    #    forward-filled between actual funding events, so a rolling std mostly
    #    captures the rate-of-change between events.
    if "fundingRate" in out.columns:
        fr = out["fundingRate"].astype(float)
        out["funding_rate_std_24h"] = fr.rolling(window=DAY_BARS).std()
    else:
        out["funding_rate_std_24h"] = np.nan

    # 5) Open interest change rate over 24h.
    if "openInterest" in out.columns:
        oi = out["openInterest"].astype(float)
        prev_oi = oi.shift(DAY_BARS).replace(0.0, np.nan)
        out["oi_change_rate_24h"] = (oi / prev_oi) - 1.0
    else:
        out["oi_change_rate_24h"] = np.nan

    # 6) Regime indicator: short-term RV / daily RV ratio.
    if "rv_gk_15min" in out.columns:
        rv_15 = out["rv_gk_15min"].astype(float)
        out["rv_ratio_15min_to_daily"] = rv_15 / out["rv_gk_1440min"].clip(lower=1e-12)
    else:
        out["rv_ratio_15min_to_daily"] = np.nan

    out[LONG_RANGE_COLS] = (
        out[LONG_RANGE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .bfill()
        .ffill()
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add 6 long-range features to the RV dataset.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=None,
                        help="Defaults to overwriting --input in place.")
    return parser.parse_args()


def _progress(step: int, total: int, message: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {message}")


def main() -> None:
    total_steps = 4
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path

    _progress(1, total_steps, f"Loading CSV: {input_path}")
    df = pd.read_csv(input_path)

    _progress(2, total_steps, f"Computing {len(LONG_RANGE_COLS)} long-range features")
    out = add_long_range_features(df)

    _progress(3, total_steps, f"Saving CSV: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    _progress(4, total_steps, "Done")
    print(f"Saved: {output_path}")
    print(f"Shape: {out.shape}")
    diag = out[LONG_RANGE_COLS].describe().T[["count", "mean", "std", "min", "max"]]
    print("\nLong-range feature stats:")
    print(diag.to_string())


if __name__ == "__main__":
    main()
