"""
Оркестратор полного формирования датасета 5m (Bybit).

По умолчанию работает с BTCUSDT, но через флаг ``--symbol ETHUSDT`` (или
любой другой Bybit-символ вида ``XXXUSDT``) перестраивает все промежуточные
и финальные пути под выбранный актив. Конкретные имена/папки берутся из
:mod:`dataset.get_data._paths` (SYMBOL читается из окружения, поэтому здесь
мы ставим ``os.environ["SYMBOL"]`` ДО запуска дочерних шагов).

Рабочая директория при запуске: каталог ``dataset/`` (родитель ``get_data/``),
чтобы относительные пути вида ``get_data/output/...`` в дочерних скриптах
совпадали.

Usage (из корня репозитория или из ``dataset/``):
  python get_data/_get_data.py
  python get_data/_get_data.py --symbol ETHUSDT
  python get_data/_get_data.py --symbol ETHUSDT --skip-fetch
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


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=DATASET_ROOT, env=env)


def _ensure_dirs(paths_mod) -> None:
    for rel in (paths_mod.CLEAN_DIR, paths_mod.INTERMEDIATE, paths_mod.FINAL_DIR, paths_mod.MAIN):
        Path(DATASET_ROOT / rel).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full dataset build orchestrator (Bybit 5m OHLCV pipeline).")
    p.add_argument(
        "--symbol",
        default=os.environ.get("SYMBOL", "BTCUSDT"),
        help="Bybit symbol (e.g. BTCUSDT, ETHUSDT, SOLUSDT). Defaults to env SYMBOL or BTCUSDT.",
    )
    p.add_argument("--start", default="2020-01-01", help="Range start (UTC if no tz)")
    p.add_argument("--end", default="2026-01-01", help="Range end (UTC if no tz)")
    p.add_argument("--interval", default="5", help="Kline interval (5 = 5m)")
    p.add_argument("--skip-fetch", action="store_true", help="Skip steps 1–2 (use existing API + clean parquet)")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    symbol = (args.symbol or "BTCUSDT").strip().upper()
    os.environ["SYMBOL"] = symbol

    # Импорт ПОСЛЕ установки os.environ["SYMBOL"], чтобы _paths использовал
    # актуальный символ при инициализации модульных констант.
    # sys.path[0] == директория этого скрипта (dataset/get_data/), поэтому
    # ``import _paths`` находит соседний модуль независимо от текущего cwd.
    sys.path.insert(0, str(_GET_DATA_DIR))
    import _paths as paths  # type: ignore[import-not-found]

    py = sys.executable
    dry = args.dry_run

    print(f"[INFO] Active symbol: {paths.SYMBOL} (lower={paths.SYMBOL_LOWER}, ccxt={paths.CCXT_SYMBOL})")

    _ensure_dirs(paths)

    # Окружение для всех дочерних процессов: пробрасываем SYMBOL.
    child_env = os.environ.copy()
    child_env["SYMBOL"] = symbol

    # --- 1) get_OHLCV: spot и perp ---
    if not args.skip_fetch:
        _run(
            [
                py,
                "get_data/get_OHLCV.py",
                "--symbol",
                symbol,
                "--category",
                "spot",
                "--interval",
                args.interval,
                "--start",
                args.start,
                "--end",
                args.end,
                "--out",
                paths.RAW_API,
            ],
            dry_run=dry,
            env=child_env,
        )
        _run(
            [
                py,
                "get_data/get_OHLCV.py",
                "--symbol",
                symbol,
                "--category",
                "linear",
                "--interval",
                args.interval,
                "--start",
                args.start,
                "--end",
                args.end,
                "--out",
                paths.RAW_PERP,
            ],
            dry_run=dry,
            env=child_env,
        )

        # --- 2) validate_OHLCV: spot и perp → clean ---
        _run(
            [py, "get_data/validate_OHLCV.py", paths.RAW_API, paths.SPOT_CLEAN],
            dry_run=dry,
            env=child_env,
        )
        _run(
            [py, "get_data/validate_OHLCV.py", paths.RAW_PERP, paths.PERP_CLEAN],
            dry_run=dry,
            env=child_env,
        )

    # --- 3) merge ---
    _run([py, "get_data/merge_dataset.py"], dry_run=dry, env=child_env)

    # --- 4–9) признаки и таргеты ---
    for script in (
        "get_data/add_time.py",
        "get_data/add_log.py",
        "get_data/add_volatility.py",
        "get_data/add_volume_statistics.py",
        "get_data/add_funding.py",
        "get_data/add_target.py",
    ):
        _run([py, script], dry_run=dry, env=child_env)

    # --- 10) финальная валидация: intermediate → _final cleaned ---
    _run(
        [
            py,
            "get_data/validate_OHLCV.py",
            paths.INTERMEDIATE_FINAL_WITH_TARGETS,
            paths.FINAL_CLEANED_PARQUET,
        ],
        dry_run=dry,
        env=child_env,
    )

    # --- 11) parquet → csv ---
    _run(
        [py, "get_data/parquet_to_csv.py", paths.FINAL_CLEANED_PARQUET, paths.FINAL_CLEANED_CSV],
        dry_run=dry,
        env=child_env,
    )

    if not dry:
        print(f"\n[DONE] Final parquet: {DATASET_ROOT / paths.FINAL_CLEANED_PARQUET}")
        print(f"[DONE] Final CSV:     {DATASET_ROOT / paths.FINAL_CLEANED_CSV}")


if __name__ == "__main__":
    main()
