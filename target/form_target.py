from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "get_data", "output", "_main", "_final", "btcusdt_5m_final_cleaned.parquet"
)
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "target", "btcusdt_5m_final_with_targets.csv")
OUTPUT_PARQUET = os.path.join(PROJECT_ROOT, "target", "btcusdt_5m_final_with_targets.parquet")


def main() -> None:
    h_minutes = 60
    step_minutes = 5
    threshold_flat = 0.001
    h_steps = h_minutes // step_minutes

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").set_index("ts")
    else:
        df = df.sort_index()

    if "close_perp" not in df.columns:
        raise ValueError("Column 'close_perp' is required.")

    future_col = f"future_close_{h_minutes}min"
    ret_col = f"delta_log_{h_minutes}min"
    df[future_col] = df["close_perp"].shift(-h_steps)
    df[ret_col] = np.log(df[future_col] / df["close_perp"])

    # Базовые таргеты
    df["base_regression"] = df[ret_col]
    df["base_class"] = "flat"
    df.loc[df["base_regression"] > threshold_flat, "base_class"] = "up"
    df.loc[df["base_regression"] < -threshold_flat, "base_class"] = "down"

    # Торгуемые таргеты по сценариям издержек
    c_scenarios = {"optimistic": 0.0010, "base": 0.0019, "pessimistic": 0.0020}
    for name, c in c_scenarios.items():
        class_col = f"trading_class_{name}"
        df[class_col] = "flat"
        df.loc[df["base_regression"] > c, class_col] = "long"
        df.loc[df["base_regression"] < -c, class_col] = "short"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.reset_index().to_csv(OUTPUT_CSV, index=False)
    df.reset_index().to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved parquet: {OUTPUT_PARQUET}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()