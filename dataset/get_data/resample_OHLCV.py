# resample_OHLCV.py
"""
Resample 1m OHLCV into higher timeframes (5m, 15m, 30m, 1h, 4h).

Input:
  - btcusdt_1m_clean.parquet (columns: ts, open, high, low, close, volume, turnover)
    ts must be UTC tz-aware.

Output:
  - 5 separate datasets (parquet or csv):
      *btc*_{5m,15m,30m,1h,4h}.parquet (or .csv)

Resampling rules:
  open    = first
  high    = max
  low     = min
  close   = last
  volume  = sum
  turnover= sum

Time labeling:
  - label="right", closed="right"
  - output ts is the RIGHT edge (window close time), safe for asof backward joins.

  python get_data/resample_OHLCV.py --in get_data/output/btcusdt_1m_clean.parquet --out_dir get_data/output/                                            
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

REQUIRED_COLS = ["ts", "open", "high", "low", "close", "volume", "turnover"]

TF_MAP = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
}


def _load_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=["ts"])
    else:
        raise ValueError("Unsupported input format. Use .parquet or .csv")

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if df["ts"].dt.tz is None:
        raise ValueError("ts must be timezone-aware (UTC).")

    # Ensure stable dtypes
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    df = df.sort_values("ts").reset_index(drop=True)
    return df


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Use ts as index for resample
    x = df_1m.set_index("ts")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "turnover": "sum",
    }

    out = (
        x.resample(rule, label="right", closed="right")
         .agg(agg)
         .dropna(subset=["open", "high", "low", "close"])  # remove empty bins
         .reset_index()
    )

    # Keep same column order
    out = out[REQUIRED_COLS]
    return out


def _save_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported output format. Use .parquet or .csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Resample 1m OHLCV into higher timeframes.")
    p.add_argument("--in", dest="in_path", required=True, help="Input 1m file (.parquet or .csv)")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--prefix", default="btcusdt", help="Output file prefix (default: btcusdt)")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output file format")
    args = p.parse_args()

    df = _load_df(args.in_path)

    print(f"Loaded 1m: {len(df):,} rows")
    print(f"Range: {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}")

    for tf, rule in TF_MAP.items():
        df_tf = resample_ohlcv(df, rule)
        out_path = os.path.join(args.out_dir, f"{args.prefix}_{tf}.{args.format}")
        _save_df(df_tf, out_path)

        # quick report
        step_mode = df_tf["ts"].diff().mode()
        print(f"[{tf}] rows={len(df_tf):,}  range={df_tf['ts'].iloc[0]} -> {df_tf['ts'].iloc[-1]}  step_mode={step_mode.iloc[0] if len(step_mode) else step_mode}")

    print("Done.")


if __name__ == "__main__":
    main()
