"""
Оркестратор полного формирования датасета BTCUSDT 5m (Bybit).

Рабочая директория при запуске: каталог `dataset/` (родитель этой папки `get_data/`),
чтобы относительные пути вида `get_data/output/...` в дочерних скриптах совпадали.

Usage (из корня репозитория или из `dataset/`):
  python get_data/_get_data.py
  python get_data/_get_data.py --skip-fetch          # пропустить шаги 1–2 (уже есть raw/clean)
  python get_data/_get_data.py --dry-run

Требования: pybit, pandas, pyarrow, ccxt (для add_funding).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Корень `dataset/`: родитель каталога `get_data/`
_GET_DATA_DIR = Path(__file__).resolve().parent
DATASET_ROOT = _GET_DATA_DIR.parent

# Относительные пути от DATASET_ROOT (как в merge_dataset, add_time, …)
MAIN = "get_data/output/_main"
RAW_API = f"{MAIN}/btcusdt_5m_spot_API.parquet"
RAW_PERP = f"{MAIN}/btcusdt_5m_perp_API.parquet"
CLEAN_DIR = f"{MAIN}/clean"
SPOT_CLEAN = f"{CLEAN_DIR}/btcusdt_5m_spot_API_clean.parquet"
PERP_CLEAN = f"{CLEAN_DIR}/btcusdt_5m_perp_API_clean.parquet"
INTERMEDIATE = f"{MAIN}/intermediate"
FINAL_DIR = f"{MAIN}/_final"
FINAL_CLEANED_PARQUET = f"{FINAL_DIR}/btcusdt_5m_final_cleaned.parquet"
FINAL_CLEANED_CSV = f"{FINAL_DIR}/btcusdt_5m_final_cleaned.csv"


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=DATASET_ROOT)


def _ensure_dirs() -> None:
    for rel in (CLEAN_DIR, INTERMEDIATE, FINAL_DIR, MAIN):
        Path(DATASET_ROOT / rel).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full dataset build orchestrator (Bybit 5m OHLCV pipeline).")
    p.add_argument("--symbol", default="BTCUSDT", help="Bybit symbol")
    p.add_argument("--start", default="2020-01-01", help="Range start (UTC if no tz)")
    p.add_argument("--end", default="2026-01-01", help="Range end (UTC if no tz)")
    p.add_argument("--interval", default="5", help="Kline interval (5 = 5m)")
    p.add_argument("--skip-fetch", action="store_true", help="Skip steps 1–2 (use existing API + clean parquet)")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    dry = args.dry_run

    _ensure_dirs()

    # --- 1) get_OHLCV: spot и perp ---
    if not args.skip_fetch:
        _run(
            [
                py,
                "get_data/get_OHLCV.py",
                "--symbol",
                args.symbol,
                "--category",
                "spot",
                "--interval",
                args.interval,
                "--start",
                args.start,
                "--end",
                args.end,
                "--out",
                RAW_API,
            ],
            dry_run=dry,
        )
        _run(
            [
                py,
                "get_data/get_OHLCV.py",
                "--symbol",
                args.symbol,
                "--category",
                "linear",
                "--interval",
                args.interval,
                "--start",
                args.start,
                "--end",
                args.end,
                "--out",
                RAW_PERP,
            ],
            dry_run=dry,
        )

        # --- 2) validate_OHLCV: spot и perp → clean ---
        _run([py, "get_data/validate_OHLCV.py", RAW_API, SPOT_CLEAN], dry_run=dry)
        _run([py, "get_data/validate_OHLCV.py", RAW_PERP, PERP_CLEAN], dry_run=dry)

    # --- 3) merge ---
    _run([py, "get_data/merge_dataset.py"], dry_run=dry)

    # --- 4–9) признаки и таргеты ---
    for script in (
        "get_data/add_time.py",
        "get_data/add_log.py",
        "get_data/add_volatility.py",
        "get_data/add_volume_statistics.py",
        "get_data/add_funding.py",
        "get_data/add_target.py",
    ):
        _run([py, script], dry_run=dry)

    # --- 10) финальная валидация: intermediate → _final cleaned ---
    intermediate_final = f"{INTERMEDIATE}/btcusdt_5m_final_with_targets.parquet"
    _run(
        [py, "get_data/validate_OHLCV.py", intermediate_final, FINAL_CLEANED_PARQUET],
        dry_run=dry,
    )

    # --- 11) parquet → csv ---
    _run(
        [py, "get_data/parquet_to_csv.py", FINAL_CLEANED_PARQUET, FINAL_CLEANED_CSV],
        dry_run=dry,
    )

    if not dry:
        print(f"\n[DONE] Final parquet: {DATASET_ROOT / FINAL_CLEANED_PARQUET}")
        print(f"[DONE] Final CSV:     {DATASET_ROOT / FINAL_CLEANED_CSV}")


if __name__ == "__main__":
    main()
