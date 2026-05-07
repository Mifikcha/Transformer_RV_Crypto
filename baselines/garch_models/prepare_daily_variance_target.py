from __future__ import annotations

"""
Prepare a **separate target** for GARCH-family models: daily return variance.

Input:
  target/{symbol}_5m_final_with_targets.(parquet|csv)  (produced by target/form_target.py)

Output (under baselines/garch_models/data/<symbol_lower>/):
  - daily_variance.parquet
  - daily_variance.csv
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.garch_models._common import (
    DailyVarianceDataset,
    data_dir,
    get_symbol_lower,
    load_5m_with_targets,
    make_daily_variance_dataset,
    resolve_5m_targets_path,
)


def build(symbol_lower: str) -> tuple[DailyVarianceDataset, Path, Path]:
    inp = resolve_5m_targets_path(symbol_lower)
    df_5m = load_5m_with_targets(inp)
    daily_ds = make_daily_variance_dataset(df_5m)
    out_dir = data_dir(symbol_lower)
    out_parquet = out_dir / "daily_variance.parquet"
    out_csv = out_dir / "daily_variance.csv"
    return daily_ds, out_parquet, out_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol-lower", type=str, default=None, help="Override symbol (lowercase).")
    args = p.parse_args()

    symbol_lower = (args.symbol_lower or get_symbol_lower()).strip().lower()
    daily_ds, out_parquet, out_csv = build(symbol_lower)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    daily_ds.df.to_parquet(out_parquet, index=False)
    daily_ds.df.to_csv(out_csv, index=False)

    print(f"[OK] Saved: {out_parquet}")
    print(f"[OK] Saved: {out_csv}")
    print(f"[INFO] Rows: {len(daily_ds.df):,}")
    print(f"[INFO] Columns: {list(daily_ds.df.columns)}")
    print(f"[INFO] ts range: {daily_ds.df['ts'].min()} -> {daily_ds.df['ts'].max()}")


if __name__ == "__main__":
    main()

