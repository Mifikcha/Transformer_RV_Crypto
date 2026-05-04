from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_ROOT / "output"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_sync_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/rv_bot")
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "+psycopg2")
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def build_engine():
    return create_engine(get_sync_database_url(), pool_pre_ping=True)


def _get_columns(conn, table: str) -> set[str]:
    q = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :table_name
        """
    )
    rows = conn.execute(q, {"table_name": table}).fetchall()
    return {str(r[0]) for r in rows}


def load_base_frame(limit_rows: int | None = None) -> pd.DataFrame:
    """Load aligned bars + predictions + rv_actual for spike pipeline."""
    with build_engine().connect() as conn:
        pred_cols = _get_columns(conn, "predictions")
        actual_cols = _get_columns(conn, "rv_actual")

        pred_select = []
        for c in ("rv_3bar", "rv_12bar", "rv_48bar", "rv_288bar"):
            if c in pred_cols:
                pred_select.append(f"p.{c} AS {c}_pred")
            else:
                pred_select.append(f"NULL::double precision AS {c}_pred")

        actual_select = []
        for c in ("rv_3bar", "rv_12bar", "rv_48bar", "rv_288bar"):
            if c in actual_cols:
                actual_select.append(f"a.{c} AS {c}_actual")
            else:
                actual_select.append(f"NULL::double precision AS {c}_actual")

        sql = f"""
            SELECT
                b.ts,
                b.open_perp,
                b.high_perp,
                b.low_perp,
                b.close_perp,
                b.volume_perp,
                b.turnover_perp,
                b.funding_rate,
                b.open_interest,
                {", ".join(pred_select)},
                {", ".join(actual_select)}
            FROM bars_5m b
            LEFT JOIN predictions p ON p.ts = b.ts
            LEFT JOIN rv_actual a ON a.ts = b.ts
            ORDER BY b.ts
        """
        df = pd.read_sql(text(sql), conn)

    if limit_rows and limit_rows > 0 and len(df) > limit_rows:
        df = df.iloc[-limit_rows:].copy()

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    numeric_cols = [c for c in df.columns if c != "ts"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Prefer actual RV for labeling/analysis, fallback to prediction if missing.
    df["rv_3bar"] = df["rv_3bar_actual"].fillna(df["rv_3bar_pred"])
    df["rv_12bar"] = df["rv_12bar_actual"].fillna(df["rv_12bar_pred"])
    df["rv_48bar"] = df["rv_48bar_actual"].fillna(df["rv_48bar_pred"])
    df["rv_288bar"] = df["rv_288bar_actual"].fillna(df["rv_288bar_pred"])

    # Keep column names consistent with historical dataset conventions.
    df["fundingRate"] = df["funding_rate"]
    df["openInterest"] = df["open_interest"]

    return df


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    out = ensure_output_dir() / filename
    df.to_csv(out, index=False)
    return out

