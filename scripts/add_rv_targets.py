"""
Add forward-looking realized volatility targets.

По умолчанию (v5): таргеты на основе Garman–Klass point volatility (_forward_rv_gk),
как в RV_Transformer_Plan_v5.md. Требуются колонки OHLC perp (или vol_garman_klass).

Usage:
  python scripts/add_rv_targets.py
  python scripts/add_rv_targets.py --input path/to/input.csv --output path/to/output.csv
  python scripts/add_rv_targets.py --close-only   # legacy: close-to-close RV
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd


def _garman_klass_point(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Garman-Klass (1980) point volatility; matches add_volatility.garman_klass_estimator."""
    # Protect against zeros/negatives that could create inf/nan on division/log.
    o = open_.clip(lower=1e-12)
    h = high.clip(lower=1e-12)
    l = low.clip(lower=1e-12)
    c = close.clip(lower=1e-12)

    ratio_hl = (h / l).clip(lower=1.0001, upper=1e6)
    ratio_co = (c / o).clip(lower=1e-6, upper=1e6)
    ln_hl = np.log(ratio_hl)
    ln_co = np.log(ratio_co)
    inner = 0.5 * ln_hl**2 - (2 * np.log(2) - 1) * ln_co**2
    return pd.Series(np.sqrt(np.maximum(0.0, inner)), index=c.index)


def _forward_rv_gk(gk_point: pd.Series, horizon: int) -> pd.Series:
    """Forward RV from squared GK point estimates (шаг 2 v5: OHLC-based targets)."""
    rv = np.sqrt(
        gk_point.pow(2).rolling(window=horizon, min_periods=horizon).sum()
    )
    return rv.shift(-horizon + 1)


def _forward_rv_close(log_ret: pd.Series, horizon: int) -> pd.Series:
    """Legacy close-only forward RV (для --close-only)."""
    rv = np.sqrt(log_ret.pow(2).rolling(window=horizon, min_periods=horizon).sum())
    return rv.shift(-horizon + 1)


DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "target",
    "btcusdt_5m_final_with_targets.csv",
)


def add_rv_targets(df: pd.DataFrame, *, close_only: bool = False) -> pd.DataFrame:
    out = df.copy()
    if "close_perp" not in out.columns:
        raise ValueError("Column 'close_perp' is required.")

    if close_only:
        if "log_return_1min" in out.columns:
            r = out["log_return_1min"].astype(float)
        else:
            r = np.log(
                out["close_perp"].astype(float) / out["close_perp"].astype(float).shift(1)
            )
        out["rv_3bar_fwd"] = _forward_rv_close(r, 3)
        out["rv_12bar_fwd"] = _forward_rv_close(r, 12)
        out["rv_48bar_fwd"] = _forward_rv_close(r, 48)
        out["rv_288bar_fwd"] = _forward_rv_close(r, 288)
    else:
        for col in ("open_perp", "high_perp", "low_perp"):
            if col not in out.columns:
                raise ValueError(
                    f"Column '{col}' is required for GK-based RV targets (v5). "
                    "Rebuild dataset with OHLC columns or use --close-only."
                )

        if "vol_garman_klass" in out.columns:
            gk = out["vol_garman_klass"].astype(float)
        else:
            gk = _garman_klass_point(
                out["open_perp"].astype(float),
                out["high_perp"].astype(float),
                out["low_perp"].astype(float),
                out["close_perp"].astype(float),
            )

        out["rv_3bar_fwd"] = _forward_rv_gk(gk, 3)
        out["rv_12bar_fwd"] = _forward_rv_gk(gk, 12)
        out["rv_48bar_fwd"] = _forward_rv_gk(gk, 48)
        out["rv_288bar_fwd"] = _forward_rv_gk(gk, 288)
    rv_cols = ["rv_3bar_fwd", "rv_12bar_fwd", "rv_48bar_fwd", "rv_288bar_fwd"]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=rv_cols).reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--close-only",
        action="store_true",
        help="Legacy close-to-close RV targets (pre-v5). Ignores GK.",
    )
    return parser.parse_args()


def _progress(step: int, total: int, message: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {message}")


def main() -> None:
    total_steps = 5
    step = 0
    args = parse_args()
    step += 1
    _progress(step, total_steps, "Parsed CLI arguments")
    input_path = args.input
    output_path = args.output or input_path

    step += 1
    _progress(step, total_steps, f"Loading CSV: {input_path}")
    df = pd.read_csv(input_path)

    step += 1
    _progress(step, total_steps, "Computing forward-looking RV targets")
    out = add_rv_targets(df, close_only=args.close_only)

    step += 1
    _progress(step, total_steps, f"Saving output CSV: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    step += 1
    _progress(step, total_steps, "Done")
    print(f"Saved: {output_path}")
    print(f"Shape: {out.shape}")


if __name__ == "__main__":
    main()
