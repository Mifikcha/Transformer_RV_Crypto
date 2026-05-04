from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spike_warning.common import OUTPUT_DIR, build_engine
from spike_warning.extract_features import extract_spike_features

DEFAULT_BUNDLE_PATH = OUTPUT_DIR / "spike_classifier_bundle.joblib"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "classifier_metrics.json"


def _prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    num_cols = [c for c in out.columns if c != "ts"]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")

    out["rv_3bar"] = out.get("rv_3bar_actual").fillna(out.get("rv_3bar_pred"))
    out["rv_12bar"] = out.get("rv_12bar_actual").fillna(out.get("rv_12bar_pred"))
    if "rv_48bar_actual" in out.columns or "rv_48bar_pred" in out.columns:
        out["rv_48bar"] = out.get("rv_48bar_actual").fillna(out.get("rv_48bar_pred"))
    else:
        out["rv_48bar"] = out["rv_12bar"].rolling(4, min_periods=1).mean()
    if "rv_288bar_actual" in out.columns or "rv_288bar_pred" in out.columns:
        out["rv_288bar"] = out.get("rv_288bar_actual").fillna(out.get("rv_288bar_pred"))
    else:
        out["rv_288bar"] = out["rv_12bar"].rolling(288, min_periods=12).mean()
    out["fundingRate"] = out.get("funding_rate")
    out["openInterest"] = out.get("open_interest")
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def _load_recent_sync(limit_rows: int = 12000) -> pd.DataFrame:
    sql = """
        WITH recent_bars AS (
            SELECT *
            FROM bars_5m
            ORDER BY ts DESC
            LIMIT :limit_rows
        )
        SELECT
            b.ts,
            b.open_perp, b.high_perp, b.low_perp, b.close_perp, b.volume_perp,
            b.turnover_perp, b.funding_rate, b.open_interest,
            p.rv_3bar AS rv_3bar_pred, p.rv_12bar AS rv_12bar_pred,
            a.rv_3bar AS rv_3bar_actual, a.rv_12bar AS rv_12bar_actual
        FROM recent_bars b
        LEFT JOIN predictions p ON p.ts = b.ts
        LEFT JOIN rv_actual a ON a.ts = b.ts
        ORDER BY b.ts
    """
    with build_engine().connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"limit_rows": limit_rows})
    return _prepare_base_frame(df)


async def _load_recent_async(
    session: AsyncSession,
    *,
    limit_rows: int = 12000,
    ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    where_clause = "WHERE ts <= :target_ts" if ts is not None else ""
    sql = f"""
        WITH recent_bars AS (
            SELECT *
            FROM bars_5m
            {where_clause}
            ORDER BY ts DESC
            LIMIT :limit_rows
        )
        SELECT
            b.ts,
            b.open_perp, b.high_perp, b.low_perp, b.close_perp, b.volume_perp,
            b.turnover_perp, b.funding_rate, b.open_interest,
            p.rv_3bar AS rv_3bar_pred, p.rv_12bar AS rv_12bar_pred,
            a.rv_3bar AS rv_3bar_actual, a.rv_12bar AS rv_12bar_actual
        FROM recent_bars b
        LEFT JOIN predictions p ON p.ts = b.ts
        LEFT JOIN rv_actual a ON a.ts = b.ts
        ORDER BY b.ts
    """
    params = {"limit_rows": limit_rows}
    if ts is not None:
        params["target_ts"] = ts.to_pydatetime()
    rows = (await session.execute(text(sql), params)).mappings().all()
    df = pd.DataFrame(rows)
    return _prepare_base_frame(df)


@lru_cache(maxsize=4)
def _load_bundle(bundle_path: str) -> dict:
    return joblib.load(bundle_path)


@lru_cache(maxsize=4)
def _load_threshold(metrics_path: str) -> float | None:
    p = Path(metrics_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return float(data.get("best_threshold"))
    except Exception:
        return None


def _predict_latest(df: pd.DataFrame, bundle_path: str, metrics_path: str) -> dict:
    if df.empty:
        return {"available": False, "reason": "no_data"}
    bundle_file = Path(bundle_path)
    if not bundle_file.exists():
        return {"available": False, "reason": f"model_not_found:{bundle_file}"}

    bundle = _load_bundle(str(bundle_file))
    model = bundle["model"]
    feature_cols = list(bundle["feature_columns"])
    feats = extract_spike_features(df)
    if feats.empty:
        return {"available": False, "reason": "no_features"}
    X = feats[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest = X.iloc[[-1]]
    proba = float(model.predict_proba(latest)[:, 1][0])

    threshold = _load_threshold(metrics_path) or 0.5
    medium_cut = max(threshold * 0.6, 0.3)
    if proba >= threshold:
        level = "high"
    elif proba >= medium_cut:
        level = "elevated"
    else:
        level = "low"

    return {
        "available": True,
        "probability": proba,
        "threshold": float(threshold),
        "level": level,
        "model_name": bundle.get("model_name", "unknown"),
        "asof_ts": df["ts"].iloc[-1],
    }


def get_dashboard_spike_signal(
    *,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    lookback_rows: int = 12000,
) -> dict:
    df = _load_recent_sync(limit_rows=lookback_rows)
    return _predict_latest(df, str(bundle_path), str(metrics_path))


async def get_spike_probability_async(
    session: AsyncSession,
    *,
    ts: pd.Timestamp | None = None,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    lookback_rows: int = 12000,
) -> dict:
    df = await _load_recent_async(session, limit_rows=lookback_rows, ts=ts)
    return _predict_latest(df, str(bundle_path), str(metrics_path))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Print latest spike warning probability")
    p.add_argument("--lookback-rows", type=int, default=12000)
    p.add_argument("--bundle-path", default=str(DEFAULT_BUNDLE_PATH))
    p.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = get_dashboard_spike_signal(
        bundle_path=args.bundle_path,
        metrics_path=args.metrics_path,
        lookback_rows=args.lookback_rows,
    )
    print(result)


if __name__ == "__main__":
    main()

