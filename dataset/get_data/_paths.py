"""
Symbol-aware path helpers for the dataset pipeline.

Single source of truth for every dataset/get_data/* script.
The active symbol is taken from the ``SYMBOL`` environment variable
(default ``BTCUSDT`` for backward compatibility) and used to compute
prefix-tagged file paths inside ``dataset/get_data/output/_main/``.

All paths are returned **relative** to the ``dataset/`` directory
(parent of ``get_data/``), matching the working directory used by
``_get_data.py`` when it spawns child scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_symbol() -> str:
    """Return the active trading symbol (uppercase, e.g. ``ETHUSDT``)."""
    return os.environ.get("SYMBOL", "BTCUSDT").strip().upper() or "BTCUSDT"


def get_symbol_lower() -> str:
    """Return the active symbol in lowercase (e.g. ``ethusdt``)."""
    return get_symbol().lower()


def get_ccxt_symbol() -> str:
    """Translate Bybit symbol (``ETHUSDT``) to ccxt perpetual id (``ETH/USDT:USDT``).

    Assumption: every supported symbol is a USDT-quoted linear perpetual.
    """
    sym = get_symbol()
    if sym.endswith("USDT"):
        base = sym[:-4]
        return f"{base}/USDT:USDT"
    raise ValueError(
        f"Unsupported symbol format for CCXT mapping: {sym!r}. "
        "Expected a Bybit linear perpetual symbol ending with 'USDT'."
    )


SYMBOL: str = get_symbol()
SYMBOL_LOWER: str = get_symbol_lower()
CCXT_SYMBOL: str = get_ccxt_symbol()


_DATASET_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = _DATASET_ROOT


# All paths below are relative to DATASET_ROOT.
MAIN: str = "get_data/output/_main"
CLEAN_DIR: str = f"{MAIN}/clean"
INTERMEDIATE: str = f"{MAIN}/intermediate"
FINAL_DIR: str = f"{MAIN}/_final"


# Raw OHLCV from Bybit V5 (spot + linear perpetual).
RAW_API: str = f"{MAIN}/{SYMBOL_LOWER}_5m_spot_API.parquet"
RAW_PERP: str = f"{MAIN}/{SYMBOL_LOWER}_5m_perp_API.parquet"

# Cleaned OHLCV after validate_OHLCV.py.
SPOT_CLEAN: str = f"{CLEAN_DIR}/{SYMBOL_LOWER}_5m_spot_API_clean.parquet"
PERP_CLEAN: str = f"{CLEAN_DIR}/{SYMBOL_LOWER}_5m_perp_API_clean.parquet"

# Stage outputs (each consumes the previous one).
COMBINED: str = f"{INTERMEDIATE}/{SYMBOL_LOWER}_5m_combined_2020-2026.parquet"
WITH_TIME_FEATURES: str = f"{INTERMEDIATE}/{SYMBOL_LOWER}_5m_combined_with_time_features.parquet"
WITH_PRICE_DERIVATIVES: str = f"{INTERMEDIATE}/{SYMBOL_LOWER}_5m_with_price_derivatives.parquet"
WITH_VOLATILITY: str = f"{INTERMEDIATE}/{SYMBOL_LOWER}_5m_with_volatility.parquet"
WITH_VOLUME_STATS: str = f"{INTERMEDIATE}/{SYMBOL_LOWER}_5m_with_volume_stats.parquet"
WITH_DERIVATIVES: str = f"{INTERMEDIATE}/{SYMBOL_LOWER}_5m_with_derivatives.parquet"
INTERMEDIATE_FINAL_WITH_TARGETS: str = (
    f"{INTERMEDIATE}/{SYMBOL_LOWER}_5m_final_with_targets.parquet"
)

# Final artifacts.
FINAL_CLEANED_PARQUET: str = f"{FINAL_DIR}/{SYMBOL_LOWER}_5m_final_cleaned.parquet"
FINAL_CLEANED_CSV: str = f"{FINAL_DIR}/{SYMBOL_LOWER}_5m_final_cleaned.csv"


def absolute(path: str) -> str:
    """Resolve a pipeline-relative path to an absolute path under ``DATASET_ROOT``."""
    return str(DATASET_ROOT / path)


__all__ = [
    "get_symbol",
    "get_symbol_lower",
    "get_ccxt_symbol",
    "absolute",
    "DATASET_ROOT",
    "SYMBOL",
    "SYMBOL_LOWER",
    "CCXT_SYMBOL",
    "MAIN",
    "CLEAN_DIR",
    "INTERMEDIATE",
    "FINAL_DIR",
    "RAW_API",
    "RAW_PERP",
    "SPOT_CLEAN",
    "PERP_CLEAN",
    "COMBINED",
    "WITH_TIME_FEATURES",
    "WITH_PRICE_DERIVATIVES",
    "WITH_VOLATILITY",
    "WITH_VOLUME_STATS",
    "WITH_DERIVATIVES",
    "INTERMEDIATE_FINAL_WITH_TARGETS",
    "FINAL_CLEANED_PARQUET",
    "FINAL_CLEANED_CSV",
]
