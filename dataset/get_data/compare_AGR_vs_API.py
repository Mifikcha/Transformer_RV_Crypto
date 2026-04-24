# compare_AGR_vs_API.py
"""
Compare OHLCV datasets:
- AGR: resampled from 1m
- API: downloaded directly from exchange

Assumes filenames like:
  btcusdt_5m_perp_AGR.parquet
  btcusdt_5m_perp_API.parquet
  ...

Outputs:
- Console summary per timeframe
- CSV report with all requested metrics

python get_data/compare_AGR_vs_API_GROK.py --dir get_data/output/perp/
 --symbol btcusdt --suffix perp --out report.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CORE_COLS = ["open", "high", "low", "close", "volume", "turnover"]


@dataclass
class ColReport:
    col: str
    n: int
    pct_nonzero: float
    abs_mean: float
    abs_median: float
    abs_max: float
    rel_mean: float
    rel_median: float
    rel_max: float


def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        raise ValueError(f"No 'ts' column in {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors='coerce')
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="first").reset_index(drop=True)
    return df


def align_on_ts(df_agr: pd.DataFrame, df_api: pd.DataFrame, tolerance_sec: int = 1):
    df_agr = df_agr.sort_values("ts").copy()
    df_api = df_api.sort_values("ts").copy()

    merged = pd.merge_asof(
        df_agr,
        df_api,
        on="ts",
        tolerance=pd.Timedelta(seconds=tolerance_sec),
        direction="nearest",
        suffixes=("_agr", "_api")
    )

    # оставляем только строки, где есть API-значения
    api_cols = [f"{c}_api" for c in CORE_COLS]
    merged = merged.dropna(subset=api_cols)

    a = merged[[f"{c}_agr" for c in CORE_COLS]].rename(
        columns={f"{c}_agr": c for c in CORE_COLS}
    )
    b = merged[[f"{c}_api" for c in CORE_COLS]].rename(
        columns={f"{c}_api": c for c in CORE_COLS}
    )

    return a.reset_index(drop=True), b.reset_index(drop=True)


def compute_col_report(a: pd.Series, b: pd.Series, col: str, eps: float = 1e-12) -> ColReport:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = (~a.isna()) & (~b.isna())
    a = a[mask].to_numpy(dtype=np.float64)
    b = b[mask].to_numpy(dtype=np.float64)

    if a.size == 0:
        return ColReport(col, 0, *[np.nan]*8)

    diff = a - b
    absd = np.abs(diff)
    denom = np.maximum(np.abs(b), eps)
    reld = absd / denom

    pct_nonzero = float((absd > 0).mean() * 100.0)

    return ColReport(
        col=col,
        n=int(a.size),
        pct_nonzero=pct_nonzero,
        abs_mean=float(np.mean(absd)),
        abs_median=float(np.median(absd)),
        abs_max=float(np.max(absd)),
        rel_mean=float(np.mean(reld)),
        rel_median=float(np.median(reld)),
        rel_max=float(np.max(reld)),
    )


def summarize_time_coverage(df: pd.DataFrame) -> Tuple[str, str, int]:
    if df.empty:
        return "NaT", "NaT", 0
    return (
        df["ts"].iloc[0].strftime('%Y-%m-%d %H:%M:%S%z'),
        df["ts"].iloc[-1].strftime('%Y-%m-%d %H:%M:%S%z'),
        len(df)
    )


def compare_pair(path_agr: str, path_api: str, eps: float) -> Dict:
    df_agr = load_parquet(path_agr)
    df_api = load_parquet(path_api)

    # Debug: показываем начало и конец временных меток
    print(f"[{os.path.basename(path_agr)}] AGR ts head (5):")
    print(df_agr['ts'].head(5).dt.strftime('%Y-%m-%d %H:%M:%S').tolist())
    print(f"[{os.path.basename(path_api)}] API ts head (5):")
    print(df_api['ts'].head(5).dt.strftime('%Y-%m-%d %H:%M:%S').tolist())
    print()

    agr_from, agr_to, agr_n = summarize_time_coverage(df_agr)
    api_from, api_to, api_n = summarize_time_coverage(df_api)

    # Ограничиваем API диапазоном AGR (чтобы не сравнивать лишнее)
    df_api = df_api[
        (df_api["ts"] >= df_agr["ts"].min()) &
        (df_api["ts"] <= df_agr["ts"].max())
    ]

    a_al, b_al = align_on_ts(df_agr, df_api, tolerance_sec=1)
    n_common = len(a_al)

    cols = [c for c in CORE_COLS if c in a_al.columns and c in b_al.columns]
    col_reports = [compute_col_report(a_al[c], b_al[c], c, eps=eps) for c in cols]

    close_rep = next((r for r in col_reports if r.col == "close"), None)

    # Критичность по close (можно подкрутить пороги под себя)
    critical = False
    if close_rep and close_rep.n > 0:
        price_med = float(np.median(pd.to_numeric(b_al["close"], errors="coerce").dropna()))
        if price_med > 0:
            if close_rep.abs_max > 0.001 * price_med:           # >0.1% от цены
                critical = True
            if close_rep.rel_mean > 1e-4:                       # >0.01% в среднем
                critical = True

    out = {
        "agr_path": os.path.basename(path_agr),
        "api_path": os.path.basename(path_api),
        "agr_rows": agr_n,
        "api_rows": api_n,
        "common_rows": n_common,
        "agr_range_from": agr_from,
        "agr_range_to": agr_to,
        "api_range_from": api_from,
        "api_range_to": api_to,
        "is_critical_by_close": critical,
    }

    # Добавляем все метрики в требуемом формате: <field>_<metric>
    for r in col_reports:
        prefix = r.col
        out[f"{prefix}_pct_nonzero"]  = r.pct_nonzero
        out[f"{prefix}_abs_mean"]     = r.abs_mean
        out[f"{prefix}_abs_median"]   = r.abs_median
        out[f"{prefix}_abs_max"]      = r.abs_max
        out[f"{prefix}_rel_mean"]     = r.rel_mean
        out[f"{prefix}_rel_median"]   = r.rel_median
        out[f"{prefix}_rel_max"]      = r.rel_max

    return out


def find_pairs(directory: str, symbol: str, suffix: str, tfs: List[str]) -> List[Tuple[str, str, str]]:
    pairs = []
    for tf in tfs:
        agr = os.path.join(directory, f"{symbol}_{tf}_{suffix}_AGR.parquet")
        api = os.path.join(directory, f"{symbol}_{tf}_{suffix}_API.parquet")
        if os.path.exists(agr) and os.path.exists(api):
            pairs.append((tf, agr, api))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Compare AGR(resampled) vs API(exchange) OHLCV parquet files.")
    p.add_argument("--dir", required=True, help="Directory with parquet files")
    p.add_argument("--symbol", default="btcusdt", help="Symbol prefix")
    p.add_argument("--suffix", default="perp", help="Middle suffix")
    p.add_argument("--tfs", default="5m,15m,30m,1h,4h", help="Timeframes")
    p.add_argument("--eps", type=float, default=1e-12, help="Epsilon")
    p.add_argument("--out", default="report.csv", help="Output CSV")
    args = p.parse_args()

    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    pairs = find_pairs(args.dir, args.symbol, args.suffix, tfs)

    if not pairs:
        raise SystemExit("No matching AGR/API pairs found.")

    reports = []
    print("\n=== AGR vs API COMPARISON ===")
    print(f"dir={args.dir} symbol={args.symbol} suffix={args.suffix}")
    print(f"timeframes: {', '.join([tf for tf,_,_ in pairs])}\n")

    for tf, agr_path, api_path in pairs:
        rep = compare_pair(agr_path, api_path, eps=args.eps)
        reports.append(rep)

        price_note = ""
        if "close_abs_max" in rep:
            price_note = f"close_abs_max={rep['close_abs_max']:.6g}, close_rel_mean={rep['close_rel_mean']:.6g}"

        print(f"[{tf}] common_rows={rep['common_rows']:,}  critical={rep['is_critical_by_close']}")
        print(f"     AGR rows={rep['agr_rows']:,} range={rep['agr_range_from']} → {rep['agr_range_to']}")
        print(f"     API rows={rep['api_rows']:,} range={rep['api_range_from']} → {rep['api_range_to']}")
        print(f"     {price_note}")
        if "volume_abs_max" in rep:
            print(f"     volume_abs_max={rep['volume_abs_max']:.6g}, volume_rel_mean={rep['volume_rel_mean']:.6g}")
        print()

    df_report = pd.DataFrame(reports)

    print("=== OVERALL ===")
    crit_n = int(df_report["is_critical_by_close"].sum())
    print(f"Critical-by-close timeframes: {crit_n}/{len(df_report)}")
    print("Tip: if close_abs_max is tiny and close_rel_mean ~0 → differences are negligible.\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df_report.to_csv(args.out, index=False)
        print(f"Saved report to: {args.out}")


if __name__ == "__main__":
    main()