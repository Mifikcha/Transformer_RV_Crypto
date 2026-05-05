from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from tg_bot.feature_engine import FEATURE_COLS, FeatureEngine


def main() -> int:
    path = "target/btcusdt_5m_final_with_targets.csv"
    df_hist = (
        pd.read_csv(path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    )

    engine = FeatureEngine(buffer_size=7000, min_bars=250)
    results: list[dict] = []
    for _, row in df_hist.iterrows():
        engine.add_bar(row.to_dict())
        feats = engine.compute_features()
        if feats is not None:
            results.append({"ts": row["ts"], **feats})
        if len(results) >= 2000:
            break

    if not results:
        print("FAIL: compute_features() never returned features")
        return 2

    df_engine = pd.DataFrame(results).set_index("ts")
    df_hist_idx = df_hist.set_index("ts")
    overlap = df_engine.index.intersection(df_hist_idx.index)
    if overlap.empty:
        print("FAIL: no overlap between engine output and history")
        return 3

    tol = 1e-5
    failed: list[str] = []
    skipped: list[str] = []
    for feat in FEATURE_COLS:
        if feat not in df_hist_idx.columns:
            skipped.append(feat)
            continue
        mae = np.abs(df_engine.loc[overlap, feat] - df_hist_idx.loc[overlap, feat]).mean()
        if float(mae) > tol:
            failed.append(f"{feat}: MAE={mae:.2e}")

    if skipped:
        print(f"SKIP: {len(skipped)} features missing in historical CSV")
    if failed:
        print("FAIL: feature mismatch")
        for line in failed:
            print("  " + line)
        return 4

    print(f"OK: all {len(FEATURE_COLS) - len(skipped)} features match (MAE <= {tol})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

