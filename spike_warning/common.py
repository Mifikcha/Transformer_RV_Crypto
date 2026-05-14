from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, ProgrammingError

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_ROOT / "output"


def mirror_repo_csv(path: Path | str) -> None:
    """Copy CSV into scripts/output and ~deep_lom/.../Графики/data (requires repo root on sys.path)."""
    repo_root = PACKAGE_ROOT.parent
    rs = str(repo_root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    from scripts.experiment_outputs import mirror_saved_csv

    mirror_saved_csv(path)


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


def _default_spike_csv_path() -> Path:
    """Same convention as transformer: target/<symbol>_5m_final_with_targets.csv."""
    repo = PACKAGE_ROOT.parent
    sym = os.environ.get("SYMBOL", "BTCUSDT").strip().lower() or "btcusdt"
    return (repo / "target" / f"{sym}_5m_final_with_targets.csv").resolve()


def _finalize_base_frame(df: pd.DataFrame, limit_rows: int | None) -> pd.DataFrame:
    if limit_rows and limit_rows > 0 and len(df) > limit_rows:
        df = df.iloc[-limit_rows:].copy()

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    numeric_cols = [c for c in df.columns if c != "ts"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Prefer actual RV for labeling/analysis, fallback to prediction if missing.
    return df.assign(
        rv_3bar=df["rv_3bar_actual"].fillna(df["rv_3bar_pred"]),
        rv_12bar=df["rv_12bar_actual"].fillna(df["rv_12bar_pred"]),
        rv_48bar=df["rv_48bar_actual"].fillna(df["rv_48bar_pred"]),
        rv_288bar=df["rv_288bar_actual"].fillna(df["rv_288bar_pred"]),
        fundingRate=df["funding_rate"],
        openInterest=df["open_interest"],
    )


def load_base_frame_from_csv(
    csv_path: Path | str | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    """Load bars + RV from local CSV when PostgreSQL bars_5m is unavailable."""
    p = Path(csv_path).expanduser().resolve() if csv_path else Path(
        os.environ.get("SPIKE_DATA_CSV", "").strip() or _default_spike_csv_path()
    )
    if not p.is_file():
        raise FileNotFoundError(
            f"SPIKE_DATA_CSV / default target CSV not found: {p}. "
            "Set SPIKE_DATA_CSV to your *_5m_final_with_targets.csv or run with a populated DB."
        )
    df = pd.read_csv(p)
    extra: dict[str, pd.Series] = {}
    if "funding_rate" not in df.columns:
        extra["funding_rate"] = df["fundingRate"] if "fundingRate" in df.columns else pd.Series(float("nan"), index=df.index)
    if "open_interest" not in df.columns:
        extra["open_interest"] = df["openInterest"] if "openInterest" in df.columns else pd.Series(float("nan"), index=df.index)

    for src, dst in (
        ("rv_3bar_fwd", "rv_3bar_actual"),
        ("rv_12bar_fwd", "rv_12bar_actual"),
        ("rv_48bar_fwd", "rv_48bar_actual"),
        ("rv_288bar_fwd", "rv_288bar_actual"),
    ):
        if dst not in df.columns:
            extra[dst] = df[src] if src in df.columns else pd.Series(float("nan"), index=df.index)
    for name in ("rv_3bar_pred", "rv_12bar_pred", "rv_48bar_pred", "rv_288bar_pred"):
        if name not in df.columns:
            extra[name] = pd.Series(float("nan"), index=df.index)
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

    print(f"[spike_warning] load_base_frame: using local CSV ({p.name}, n={len(df):,})")
    return _finalize_base_frame(df, limit_rows)


def load_base_frame(limit_rows: int | None = None) -> pd.DataFrame:
    """Load aligned bars + predictions + rv_actual for spike pipeline.

    Tries PostgreSQL (bars_5m + joins). If tables are missing or SPIKE_USE_LOCAL_CSV=1,
    falls back to target/<symbol>_5m_final_with_targets.csv (override with SPIKE_DATA_CSV).
    """
    use_csv = os.environ.get("SPIKE_USE_LOCAL_CSV", "").strip().lower() in {"1", "true", "yes"}
    if use_csv:
        return load_base_frame_from_csv(None, limit_rows)

    df: pd.DataFrame | None = None
    try:
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
    except (ProgrammingError, DBAPIError, OSError) as exc:
        msg = str(exc).strip().split("\n", 1)[0]
        if len(msg) > 180:
            msg = msg[:177] + "..."
        print(f"[spike_warning] load_base_frame: DB unavailable ({msg}); falling back to CSV.")
        return load_base_frame_from_csv(None, limit_rows)

    if df is None:
        return load_base_frame_from_csv(None, limit_rows)

    print("[spike_warning] load_base_frame: using PostgreSQL (bars_5m + joins)")
    return _finalize_base_frame(df, limit_rows)


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    out = ensure_output_dir() / filename
    df.to_csv(out, index=False)
    mirror_repo_csv(out)
    return out

