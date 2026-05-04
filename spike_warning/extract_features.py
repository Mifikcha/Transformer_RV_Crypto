from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from spike_warning.common import OUTPUT_DIR, load_base_frame, save_csv


def _regime_duration(mask: pd.Series) -> pd.Series:
    dur = mask.groupby((~mask).cumsum()).cumsum()
    return dur.where(mask, 0).astype(float)


def extract_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Group 1: current volatility level/state.
    out["rv_3bar_current"] = df["rv_3bar"]
    out["rv_12bar_current"] = df["rv_12bar"]
    rv_mean_24h = df["rv_3bar"].rolling(288).mean()
    rv_std_24h = df["rv_3bar"].rolling(288).std()
    out["rv_zscore_24h"] = (df["rv_3bar"] - rv_mean_24h) / rv_std_24h.clip(lower=1e-12)
    out["rv_percentile_30d"] = df["rv_3bar"].rolling(8640, min_periods=2880).rank(pct=True)

    # Group 2: regime and regime duration.
    p25 = df["rv_3bar"].rolling(8640, min_periods=2880).quantile(0.25)
    p75 = df["rv_3bar"].rolling(8640, min_periods=2880).quantile(0.75)
    out["is_low_regime"] = (df["rv_3bar"] <= p25).astype(int)
    out["is_high_regime"] = (df["rv_3bar"] >= p75).astype(int)
    out["low_regime_duration"] = _regime_duration(df["rv_3bar"] <= p25)
    out["high_regime_duration"] = _regime_duration(df["rv_3bar"] >= p75)

    # Group 3: term structure.
    out["term_slope"] = df["rv_3bar"] / df["rv_288bar"].clip(lower=1e-12)
    out["term_inverted"] = (out["term_slope"] > 1.0).astype(int)

    # Group 4: funding and OI.
    fr = df["fundingRate"]
    fr_mean = fr.rolling(8640, min_periods=2880).mean()
    fr_std = fr.rolling(8640, min_periods=2880).std()
    out["funding_zscore"] = (fr - fr_mean) / fr_std.clip(lower=1e-12)
    out["funding_abs_zscore"] = out["funding_zscore"].abs()
    oi = df["openInterest"]
    out["oi_change_1h"] = oi.pct_change(12)
    out["oi_change_4h"] = oi.pct_change(48)
    out["oi_abs_change_4h"] = out["oi_change_4h"].abs()

    # Group 5: volume anomaly.
    vol = df["volume_perp"]
    vol_median_7d = vol.rolling(2016, min_periods=576).median()
    out["volume_ratio_7d"] = vol / vol_median_7d.clip(lower=1e-12)
    vol_1h = vol.rolling(12).sum()
    vol_prev_1h = vol.shift(12).rolling(12).sum()
    out["volume_acceleration"] = vol_1h / vol_prev_1h.clip(lower=1e-12)

    # Group 6: price dynamics.
    close = df["close_perp"]
    out["abs_return_1h"] = close.pct_change(12).abs()
    out["abs_return_4h"] = close.pct_change(48).abs()
    rolling_high_4h = df["high_perp"].rolling(48).max()
    rolling_low_4h = df["low_perp"].rolling(48).min()
    out["range_4h"] = (rolling_high_4h - rolling_low_4h) / close.clip(lower=1.0)

    # Group 7: session/time features.
    ts = pd.to_datetime(df["ts"], utc=True)
    out["hour"] = ts.dt.hour
    out["is_ny_session"] = out["hour"].between(13, 21).astype(int)
    out["is_asia_session"] = out["hour"].between(0, 8).astype(int)

    # Group 8: RV dynamics.
    out["rv_change_1h"] = df["rv_3bar"].pct_change(12)
    out["rv_change_4h"] = df["rv_3bar"].pct_change(48)
    rv_diff = df["rv_3bar"].diff()
    out["rv_acceleration"] = rv_diff.diff()

    # Metadata for downstream charts/evaluation (not model features).
    out["_meta_close_perp"] = close

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract spike warning classifier features")
    p.add_argument(
        "--labels-path",
        default=str(OUTPUT_DIR / "spike_labels.csv"),
        help="CSV from define_spikes.py",
    )
    p.add_argument("--limit-rows", type=int, default=0, help="0 = full history")
    return p


def main() -> None:
    args = build_parser().parse_args()
    labels_df = pd.read_csv(args.labels_path)
    labels_df["ts"] = pd.to_datetime(labels_df["ts"], utc=True)
    label_cols = [
        "ts",
        "spike_in_next_4h",
        "spike_event",
        "spike_threshold",
        "max_rv_next_4h",
        "time_to_spike",
    ]
    missing_required = [c for c in ("ts", "spike_in_next_4h", "spike_event") if c not in labels_df.columns]
    if missing_required:
        raise RuntimeError(f"Labels file is missing required columns: {missing_required}")
    labels_df = labels_df[[c for c in label_cols if c in labels_df.columns]].copy()

    base_df = load_base_frame(limit_rows=args.limit_rows or None)
    if base_df.empty:
        raise RuntimeError("No data loaded from DB.")

    merged = base_df.merge(labels_df, on="ts", how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)
    feats = extract_spike_features(merged)
    out = pd.concat(
        [
            merged[
                [
                    "ts",
                    "spike_in_next_4h",
                    "spike_event",
                    "spike_threshold",
                    "max_rv_next_4h",
                    "time_to_spike",
                ]
            ],
            feats,
        ],
        axis=1,
    )

    path = save_csv(out, "spike_features.csv")
    print(f"Saved: {path}")
    print(f"Rows: {len(out):,}, feature columns: {len(feats.columns):,}")


if __name__ == "__main__":
    main()

