# OHLCV.py
"""
Atomic module: Fetch FULL-range OHLCV (kline) from Bybit via pybit with correct pagination.

Why pagination works:
- Bybit returns at most 1000 klines per request (limit<=1000).
- We paginate backwards in time using `end_cursor` to retrieve all data in [start, end).

Outputs:
- Single .csv or .parquet file (chronological order, no duplicates).

Dependencies:
  pip install pybit pandas pyarrow

Examples:
  python get_data/get_OHLCV.py --symbol BTCUSDT --category linear --interval 1 --start "2020-01-01" --end "2026-01-01" --out get_data/output/perp/btcusdt_1m_perp.parquet
  python get_data/get_OHLCV.py --symbol BTCUSDT --category linear --interval 1 --start "2020-01-01" --end "2026-01-01" --out get_data/output/perp/btcusdt_1m_perp.csv
  python get_data/get_OHLCV.py --symbol BTCUSDT --category linear --interval 5 --start "2020-01-01" --end "2026-01-01" --out get_data/output/perp/btcusdt_5m_perp.parquet
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd
from pybit.unified_trading import HTTP

Category = Literal["spot", "linear", "inverse"]


# -----------------------------
# Time helpers
# -----------------------------
def _to_utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_dt(value: str) -> datetime:
    """
    Parse datetime from:
      - YYYY-MM-DD
      - YYYY-MM-DD HH:MM
      - YYYY-MM-DD HH:MM:SS
      - ISO-8601 (with or without timezone)
    If no timezone is provided, assume UTC.
    """
    ts = pd.to_datetime(value, utc=False)
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


# -----------------------------
# Core fetcher
# -----------------------------
@dataclass
class OHLCVFetcher:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = False
    max_retries: int = 6
    retry_sleep_s: float = 1.0
    rate_sleep_s: float = 0.05

    def __post_init__(self) -> None:
        self.session = HTTP(
            testnet=self.testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

    def _call_with_retries(self, fn, **kwargs):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = fn(**kwargs)
                if resp.get("retCode", None) == 0:
                    return resp
                raise RuntimeError(f"Bybit error retCode={resp.get('retCode')} retMsg={resp.get('retMsg')}")
            except Exception as e:
                last_err = e
                time.sleep(self.retry_sleep_s * attempt)
        raise RuntimeError(f"Failed after {self.max_retries} retries. Last error: {last_err}") from last_err

    def get_kline_chunk(
        self,
        symbol: str,
        category: Category,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch up to `limit` klines within [start_ms, end_ms).
        Bybit returns newest-first, we normalize to ascending order.
        Columns: ts, open, high, low, close, volume, turnover
        """
        payload = {
            "category": category,
            "symbol": symbol,
            "interval": str(interval),
            "start": start_ms,
            "end": end_ms,
            "limit": limit,
        }
        data = self._call_with_retries(self.session.get_kline, **payload)
        result = data.get("result", {}) or {}
        chunk = result.get("list", []) or []
        if not chunk:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "turnover"])

        df = pd.DataFrame(chunk, columns=["ts_ms", "open", "high", "low", "close", "volume", "turnover"])
        df["ts_ms"] = df["ts_ms"].astype("int64")
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Force stable dtypes (critical for parquet stitching)
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            df[c] = df[c].astype("float64")


        # Ascending time + dedup (just in case)
        df = df.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"], keep="last").reset_index(drop=True)
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df[["ts", "open", "high", "low", "close", "volume", "turnover"]]
        return df


# -----------------------------
# IO helpers
# -----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_chunk(df: pd.DataFrame, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in [".parquet", ".pq"]:
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Chunk extension must be .csv or .parquet/.pq")


def _concat_chunks_to_csv(chunk_paths: list[str], out_path: str) -> None:
    # Write header once, then append bodies
    with open(out_path, "wb") as out_f:
        wrote_header = False
        for p in chunk_paths:
            with open(p, "rb") as f:
                data = f.read()
            if not data:
                continue
            if not wrote_header:
                out_f.write(data)
                wrote_header = True
            else:
                # skip first line (header)
                nl = data.find(b"\n")
                if nl != -1 and nl + 1 < len(data):
                    out_f.write(data[nl + 1 :])


def _concat_chunks_to_parquet(chunk_paths: list[str], out_path: str) -> None:
    # Streaming append to a single Parquet using pyarrow ParquetWriter
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    try:
        for p in chunk_paths:
            df = pd.read_parquet(p)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


# -----------------------------
# Main pagination routine
# -----------------------------
def fetch_full_range(
    fetcher: OHLCVFetcher,
    symbol: str,
    category: Category,
    interval: str,
    start_ms: int,
    end_ms: int,
    out_path: str,
    limit: int = 1000,
    keep_temp: bool = False,
) -> None:
    """
    Paginate backwards from end_ms to start_ms, write temp chunks, then stitch in chronological order.
    """
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in [".csv", ".parquet", ".pq"]:
        raise ValueError("Unsupported output extension. Use .csv or .parquet/.pq")

    if os.path.exists(out_path):
        os.remove(out_path)

    tmp_dir = out_path + ".__tmp_chunks__"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    _ensure_dir(tmp_dir)

    end_cursor = end_ms
    chunk_files: list[str] = []
    total_rows = 0
    n_chunks = 0

    while end_cursor > start_ms:
        df = fetcher.get_kline_chunk(
            symbol=symbol,
            category=category,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_cursor,
            limit=limit,
        )

        if df.empty:
            break

        # Oldest timestamp in this chunk
        oldest_ts = df["ts"].iloc[0]
        oldest_ms = int(oldest_ts.value // 10**6)  # ns -> ms
        newest_ts = df["ts"].iloc[-1]
        newest_ms = int(newest_ts.value // 10**6)

        # Save chunk file named by oldest_ms so we can sort later
        n_chunks += 1
        chunk_path = os.path.join(tmp_dir, f"chunk_{oldest_ms}_{newest_ms}{ext}")
        _write_chunk(df, chunk_path)
        chunk_files.append(chunk_path)

        rows = len(df)
        total_rows += rows
        print(
            f"[chunk {n_chunks:05d}] rows={rows}  "
            f"range={oldest_ts} -> {newest_ts}  "
            f"end_cursor was {pd.to_datetime(end_cursor, unit='ms', utc=True)}"
        )

        # Move cursor backwards to strictly before the oldest candle
        # (avoid overlap; Bybit uses candle startTime)
        prev_end_cursor = oldest_ms
        if prev_end_cursor >= end_cursor:
            # Safety: avoid infinite loop
            break
        end_cursor = prev_end_cursor

        time.sleep(fetcher.rate_sleep_s)

    if not chunk_files:
        print("No data returned for the requested range.")
        if not keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # We fetched from newest->oldest, but chunk filenames encode oldest_ms.
    # Sort chunks by oldest_ms ascending => chronological.
    chunk_files_sorted = sorted(
        chunk_files,
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )

    print(f"Stitching {len(chunk_files_sorted)} chunks into {out_path} ...")

    if ext == ".csv":
        _concat_chunks_to_csv(chunk_files_sorted, out_path)
    else:
        _concat_chunks_to_parquet(chunk_files_sorted, out_path)

    print(f"Done. Total rows written: {total_rows}. Output: {out_path}")

    if keep_temp:
        print(f"Temp chunks kept at: {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch full-range OHLCV (kline) from Bybit via pybit.")
    p.add_argument("--symbol", required=True, help="e.g., BTCUSDT")
    p.add_argument("--category", required=True, choices=["spot", "linear", "inverse"], help="Market category")
    p.add_argument("--interval", required=True, help="Bybit kline interval: 1,3,5,15,30,60,120,240,360,720,D,W,M ...")
    p.add_argument("--start", required=True, type=str, help="Start datetime (UTC if no tz).")
    p.add_argument("--end", required=True, type=str, help="End datetime (UTC if no tz).")
    p.add_argument("--out", required=True, help="Output file (.csv or .parquet)")

    p.add_argument("--testnet", action="store_true", help="Use Bybit testnet (usually you want mainnet).")
    p.add_argument("--api-key", default=os.getenv("BYBIT_API_KEY"), help="Optional (public endpoints work without).")
    p.add_argument("--api-secret", default=os.getenv("BYBIT_API_SECRET"), help="Optional.")
    p.add_argument("--limit", type=int, default=1000, help="Kline limit per call (<=1000).")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary chunk files (debug).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    start_dt = parse_dt(args.start)
    end_dt = parse_dt(args.end)
    start_ms, end_ms = _to_utc_ms(start_dt), _to_utc_ms(end_dt)

    if end_ms <= start_ms:
        raise SystemExit("--end must be greater than --start")

    if args.limit < 1 or args.limit > 1000:
        raise SystemExit("--limit must be in [1, 1000]")

    fetcher = OHLCVFetcher(
        api_key=args.api_key,
        api_secret=args.api_secret,
        testnet=args.testnet,
    )

    fetch_full_range(
        fetcher=fetcher,
        symbol=args.symbol,
        category=args.category,
        interval=str(args.interval),
        start_ms=start_ms,
        end_ms=end_ms,
        out_path=args.out,
        limit=args.limit,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()