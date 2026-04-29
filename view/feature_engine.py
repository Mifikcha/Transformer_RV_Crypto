"""
Feature engine: reproduce all 32 features from the historical dataset pipeline.

Replicates logic from:
  - dataset/get_data/add_time.py       (time features)
  - dataset/get_data/add_log.py         (log returns)
  - dataset/get_data/add_volatility.py  (RV, SMA, ATR, OHLC estimators)
  - dataset/get_data/add_volume_statistics.py (rolling volume stats)
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    "realized_vol_240min", "openInterest", "realized_vol_60min", "sma_240min",
    "fundingRate", "sin_day", "atr_14", "weekday", "rolling_vol_std_240min",
    "sma_60min", "open_perp", "cos_day", "realized_vol_15min",
    "min_to_asia_open", "min_to_ny_open", "high_perp", "low_perp",
    "rolling_vol_mean_240min", "rv_parkinson_15min", "rv_parkinson_60min",
    "rv_parkinson_240min", "rv_gk_15min", "rv_gk_60min", "rv_gk_240min",
    "rv_rs_15min", "rv_rs_60min", "rv_rs_240min", "volume_perp",
    "turnover_perp", "volume_spot", "sin_hour", "cos_hour",
    # Present in the training checkpoint feature_columns (dataset/get_data/add_log.py & add_volatility.py)
    "cum_log_return_15min", "cum_log_return_60min", "delta_sma_15min", "delta_sma_60min",
]

# Long-horizon plan, Step 4: optional long-range features (1 day / 1 week
# windows). Not part of FEATURE_COLS by default to preserve backward
# compatibility for inference pipelines that pre-date Step 4. Live workers
# that want to use them should:
#   1) Construct ``FeatureEngine(min_bars=2200, buffer_size=>=2200)`` to cover
#      the weekly window (2016 bars + small warmup).
#   2) Pass these names via ``feature_cols`` to ``get_window``.
LONG_RANGE_FEATURE_COLS: list[str] = [
    "rv_gk_1440min",
    "rv_parkinson_1440min",
    "rv_gk_weekly",
    "funding_rate_std_24h",
    "oi_change_rate_24h",
    "rv_ratio_15min_to_daily",
]

HAR_RV_BASE = ("rv_gk_15min", "rv_gk_60min", "rv_gk_240min")
WEEK_BARS = 5 * 288   # 1440
MONTH_BARS = 22 * 288  # 6336

# Long-range windows for Step 4 features.
DAY_BARS = 288               # 1440 min  / 5 min per bar
LONG_WEEK_BARS = 7 * 288     # 2016 bars (true 7-day calendar week)

_RV_WINDOWS = [(15, 3), (60, 12), (240, 48)]
_OHLC_WINDOWS = [(3, "15min"), (12, "60min"), (48, "240min")]

_RAW_COLS = [
    "ts", "open_perp", "high_perp", "low_perp", "close_perp",
    "volume_perp", "turnover_perp", "volume_spot",
    "fundingRate", "openInterest",
]


class FeatureEngine:
    def __init__(self, buffer_size: int = 7000, min_bars: int = 250) -> None:
        self.bars: deque[dict] = deque(maxlen=buffer_size)
        self.min_bars = min_bars
        self._cache_df: pd.DataFrame | None = None
        self._cache_len: int = 0

    def add_bar(self, bar: dict) -> None:
        b = dict(bar)
        if "funding_rate" in b and "fundingRate" not in b:
            b["fundingRate"] = b.pop("funding_rate")
        if "open_interest" in b and "openInterest" not in b:
            b["openInterest"] = b.pop("open_interest")
        self.bars.append(b)
        self._cache_df = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_features(self) -> dict | None:
        """Return all 32 features for the last bar, or None if not enough data."""
        if len(self.bars) < self.min_bars:
            return None
        df = self._get_feature_df()
        last = df.iloc[-1]
        out: dict[str, float] = {}
        for col in FEATURE_COLS:
            val = last.get(col, np.nan)
            if pd.isna(val):
                return None
            out[col] = float(val)
        return out

    def compute_har_context(self) -> np.ndarray | None:
        """Return 6-element HAR context vector.

        Matches training behavior from ``dataset.add_rv_har_context_columns``:
        rolling weekly/monthly means with ``bfill().ffill()``.  When the
        buffer is too short for the monthly window the weekly mean is used
        as a fallback (much better than the previous zeros-fallback).
        """
        min_required = WEEK_BARS // 2  # 720 bars — enough for weekly mean
        if len(self.bars) < min_required:
            return None
        df = self._get_feature_df()
        har: list[float] = []
        for col in HAR_RV_BASE:
            s = df[col].astype(float)
            w_series = s.rolling(WEEK_BARS, min_periods=WEEK_BARS // 2).mean().bfill().ffill()
            m_series = s.rolling(MONTH_BARS, min_periods=MONTH_BARS // 2).mean().bfill().ffill()

            w_val = w_series.iloc[-1]
            m_val = m_series.iloc[-1]

            if pd.isna(w_val):
                return None
            if pd.isna(m_val):
                m_val = w_val
            har.extend([float(w_val), float(m_val)])
        return np.array(har, dtype=np.float64)

    def get_window(self, seq_len: int = 240, feature_cols: list[str] | None = None) -> np.ndarray | None:
        """Return [seq_len, F] feature matrix for the last *seq_len* bars."""
        if len(self.bars) < self.min_bars or len(self.bars) < seq_len:
            return None
        cols = feature_cols or FEATURE_COLS
        df = self._get_feature_df()
        if len(df) < seq_len:
            return None
        window = df[cols].iloc[-seq_len:]
        if window.isna().any().any():
            return None
        return window.values.astype(np.float64)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_feature_df(self) -> pd.DataFrame:
        if self._cache_df is not None and self._cache_len == len(self.bars):
            return self._cache_df
        df = self._build_raw_df()
        self._compute_all(df)
        self._cache_df = df
        self._cache_len = len(self.bars)
        return df

    def _build_raw_df(self) -> pd.DataFrame:
        records: list[dict] = []
        for b in self.bars:
            rec: dict = {}
            for c in _RAW_COLS:
                rec[c] = b.get(c, np.nan)
            records.append(rec)
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        for c in _RAW_COLS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @staticmethod
    def _compute_all(df: pd.DataFrame) -> None:
        c = df["close_perp"]
        h = df["high_perp"]
        l = df["low_perp"]
        o = df["open_perp"]
        v = df["volume_perp"]
        ts = df["ts"]

        # --- Time features (vectorized) --------------------------------
        hour = ts.dt.hour
        minute = ts.dt.minute
        second = ts.dt.second
        weekday = ts.dt.weekday

        df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
        df["sin_day"] = np.sin(2 * np.pi * weekday / 7)
        df["cos_day"] = np.cos(2 * np.pi * weekday / 7)
        df["weekday"] = weekday

        total_sec = hour * 3600 + minute * 60 + second
        df["min_to_asia_open"] = (86400 - total_sec) / 60

        sec_to_ny = 14 * 3600 - total_sec
        df["min_to_ny_open"] = sec_to_ny.where(sec_to_ny > 0, sec_to_ny + 86400) / 60

        # --- Log returns (bar-to-bar, = log_return_5min) ---------------
        log_ret = np.log(c / c.shift(1))
        # --- Cumulative log return windows (add_log.py) ----------------
        df["cum_log_return_15min"] = np.log(c / c.shift(3))
        df["cum_log_return_60min"] = np.log(c / c.shift(12))

        # --- Realized volatility: sqrt(sum(lr^2, window)) --------------
        lr_sq = log_ret ** 2
        for win_min, win_bars in _RV_WINDOWS:
            df[f"realized_vol_{win_min}min"] = np.sqrt(lr_sq.rolling(win_bars).sum())

        # --- SMA -------------------------------------------------------
        df["sma_60min"] = c.rolling(12).mean()
        df["sma_240min"] = c.rolling(48).mean()
        # delta SMA (add_volatility.py)
        df["delta_sma_15min"] = c.rolling(3).mean().diff()
        df["delta_sma_60min"] = df["sma_60min"].diff()

        # --- ATR (14-bar) ----------------------------------------------
        tr = np.maximum(h - l, np.maximum(np.abs(h - c.shift(1)), np.abs(l - c.shift(1))))
        df["atr_14"] = tr.rolling(14).mean()

        # --- OHLC point volatility estimators --------------------------
        ratio_hl = (h / l).clip(lower=1.0001)
        ratio_co = (c / o).clip(lower=1e-6, upper=1e6)
        ln_hl = np.log(ratio_hl)
        ln_co = np.log(ratio_co)

        vol_park = np.sqrt(ln_hl ** 2 / (4 * np.log(2)))

        inner_gk = 0.5 * ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2
        vol_gk = np.sqrt(np.maximum(0.0, inner_gk))

        hc = h.clip(lower=1e-12)
        lc = l.clip(lower=1e-12)
        oc = o.clip(lower=1e-12)
        cc = c.clip(lower=1e-12)
        inner_rs = np.log(hc / cc) * np.log(hc / oc) + np.log(lc / cc) * np.log(lc / oc)
        vol_rs = np.sqrt(np.maximum(0.0, inner_rs))

        # --- Rolling OHLC RV ------------------------------------------
        for win_bars, win_name in _OHLC_WINDOWS:
            df[f"rv_parkinson_{win_name}"] = np.sqrt(
                (vol_park ** 2).rolling(win_bars, min_periods=win_bars).sum()
            )
            df[f"rv_gk_{win_name}"] = np.sqrt(
                (vol_gk ** 2).rolling(win_bars, min_periods=win_bars).sum()
            )
            df[f"rv_rs_{win_name}"] = np.sqrt(
                (vol_rs ** 2).rolling(win_bars, min_periods=win_bars).sum()
            )

        ohlc_rv_cols = [
            c for c in df.columns
            if c.startswith("rv_parkinson_") or c.startswith("rv_gk_") or c.startswith("rv_rs_")
        ]
        df[ohlc_rv_cols] = df[ohlc_rv_cols].bfill().ffill()

        # --- Volume rolling stats (volume_perp based!) -----------------
        df["rolling_vol_mean_240min"] = v.rolling(48).mean()
        df["rolling_vol_std_240min"] = v.rolling(48).std()

        # --- Relative / stationary price features (for future retrain) ---
        # These replace absolute-price features (open_perp, high_perp, …)
        # that cause StandardScaler drift as BTC price moves away from the
        # training distribution.  Available once model is retrained.
        prev_c = c.shift(1)
        df["open_ret"] = np.log((o / prev_c).clip(lower=1e-12))
        df["high_ret"] = np.log((h / c).clip(lower=1e-12))
        df["low_ret"] = np.log((l / c).clip(lower=1e-12))
        df["close_open_ret"] = np.log((c / o).clip(lower=1e-12, upper=1e6))
        sma60 = df.get("sma_60min")
        sma240 = df.get("sma_240min")
        if sma60 is not None:
            df["sma_60min_ratio"] = c / sma60.clip(lower=1e-12)
        if sma240 is not None:
            df["sma_240min_ratio"] = c / sma240.clip(lower=1e-12)
        df["atr_14_norm"] = df.get("atr_14", 0) / c.clip(lower=1e-12)

        # --- Long-range features (plan Step 4) -------------------------
        # Daily / weekly aggregations of the existing per-bar GK and Parkinson
        # point estimators above. Same causal-rolling convention as the OHLC
        # rolling RVs; bfill/ffill applied at the end.
        df["rv_gk_1440min"] = np.sqrt(
            (vol_gk ** 2).rolling(DAY_BARS, min_periods=DAY_BARS).sum()
        )
        df["rv_parkinson_1440min"] = np.sqrt(
            (vol_park ** 2).rolling(DAY_BARS, min_periods=DAY_BARS).sum()
        )
        df["rv_gk_weekly"] = np.sqrt(
            (vol_gk ** 2).rolling(LONG_WEEK_BARS, min_periods=LONG_WEEK_BARS).sum()
        )

        funding = df.get("fundingRate")
        if funding is not None:
            df["funding_rate_std_24h"] = funding.rolling(DAY_BARS).std()
        else:
            df["funding_rate_std_24h"] = np.nan

        oi = df.get("openInterest")
        if oi is not None:
            prev_oi = oi.shift(DAY_BARS).replace(0.0, np.nan)
            df["oi_change_rate_24h"] = (oi / prev_oi) - 1.0
        else:
            df["oi_change_rate_24h"] = np.nan

        if "rv_gk_15min" in df.columns:
            df["rv_ratio_15min_to_daily"] = (
                df["rv_gk_15min"] / df["rv_gk_1440min"].clip(lower=1e-12)
            )
        else:
            df["rv_ratio_15min_to_daily"] = np.nan

        long_range_cols = [
            "rv_gk_1440min",
            "rv_parkinson_1440min",
            "rv_gk_weekly",
            "funding_rate_std_24h",
            "oi_change_rate_24h",
            "rv_ratio_15min_to_daily",
        ]
        df[long_range_cols] = (
            df[long_range_cols].replace([np.inf, -np.inf], np.nan).bfill().ffill()
        )
