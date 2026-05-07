from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s.strip())


def get_symbol() -> str:
    return os.environ.get("SYMBOL", "BTCUSDT").strip().upper() or "BTCUSDT"


def get_symbol_lower() -> str:
    return get_symbol().lower()


def resolve_5m_targets_path(symbol_lower: str) -> Path:
    """Resolve the current dataset with targets produced by target/form_target.py."""
    candidates = [
        PROJECT_ROOT / "target" / f"{symbol_lower}_5m_final_with_targets.parquet",
        PROJECT_ROOT / "target" / f"{symbol_lower}_5m_final_with_targets.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Deterministic fallback path (helps error messages stay stable).
    return candidates[0]


def output_dir(symbol_lower: str) -> Path:
    p = PROJECT_ROOT / "baselines" / "garch_models" / "output" / symbol_lower
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(symbol_lower: str) -> Path:
    p = PROJECT_ROOT / "baselines" / "garch_models" / "data" / symbol_lower
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class DailyVarianceDataset:
    df: pd.DataFrame
    returns_col: str = "ret_1d"
    variance_col: str = "var_1d"


def load_5m_with_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"5m dataset with targets not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    if "ts" not in df.columns:
        raise ValueError("Column 'ts' is required in the 5m dataset.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    if "close_perp" not in df.columns:
        raise ValueError("Column 'close_perp' is required in the 5m dataset.")
    return df


def make_daily_variance_dataset(df_5m: pd.DataFrame) -> DailyVarianceDataset:
    """Convert 5m dataset to daily close, daily log return and squared return."""
    tmp = df_5m[["ts", "close_perp"]].copy()
    tmp = tmp.set_index("ts").sort_index()
    daily_close = tmp["close_perp"].resample("1D").last().dropna()
    daily = pd.DataFrame({"ts": daily_close.index, "close": daily_close.values})
    daily["ret_1d"] = np.log(daily["close"] / daily["close"].shift(1))
    daily["var_1d"] = (daily["ret_1d"].astype(float) ** 2).astype(float)
    daily = daily.dropna().reset_index(drop=True)
    return DailyVarianceDataset(df=daily)


def metrics_for_variance(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics for positive targets (variance)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    yt_pos = np.clip(yt, eps, None)
    yp_pos = np.clip(yp, eps, None)

    mse = float(np.mean((yp_pos - yt_pos) ** 2))
    mae = float(np.mean(np.abs(yp_pos - yt_pos)))
    # R2 in log-space (same philosophy as baselines/utils.py)
    yt_log = np.log(yt_pos)
    yp_log = np.log(yp_pos)
    ss_res = float(np.sum((yt_log - yp_log) ** 2))
    ss_tot = float(np.sum((yt_log - float(np.mean(yt_log))) ** 2))
    r2_log = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    qlike = float(np.mean(np.log(yp_pos) + yt_pos / yp_pos))

    return {
        "mse": mse,
        "mae": mae,
        "r2_log": r2_log,
        "qlike": qlike,
        "n": float(len(yt_pos)),
    }


def tee_print(f: TextIO, msg: str) -> None:
    print(msg)
    f.write(msg + "\n")
    f.flush()


def ensure_arch_available() -> None:
    try:
        import arch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Package 'arch' is required for GARCH models. "
            "Install it via: pip install arch"
        ) from e


def python_exe() -> str:
    return sys.executable

