from __future__ import annotations

import os

import numpy as np
import pandas as pd

from view.feature_engine import FeatureEngine
from view.inference import RVInference


def main() -> int:
    model_path = os.getenv("MODEL_PATH", "transformer/output/models/fold_rv_4.pt")
    if not os.path.exists(model_path):
        print(f"SKIP: model file not found: {model_path}")
        return 10

    data_path = "target/btcusdt_5m_final_with_targets.csv"
    df = (
        pd.read_csv(data_path, parse_dates=["ts"])
        .sort_values("ts")
        .tail(7000)
        .reset_index(drop=True)
    )

    engine = FeatureEngine(buffer_size=7000, min_bars=250)
    for _, row in df.iterrows():
        engine.add_bar(row.to_dict())

    window = engine.get_window(seq_len=240)
    if window is None:
        print("FAIL: window is None")
        return 2

    har = engine.compute_har_context()
    if har is None:
        print("WARN: HAR context unavailable, using zeros")
        har = np.zeros(6, dtype=np.float64)

    inf = RVInference(model_path)
    out = inf.predict(window, har)
    print(f"pred: {out}")

    rv3 = out.get("rv_3bar")
    rv12 = out.get("rv_12bar")
    if rv3 is None or rv12 is None:
        print("FAIL: missing rv outputs")
        return 3

    if not (1e-8 < float(rv3) < 0.1):
        print(f"FAIL: rv_3bar out of range: {rv3}")
        return 4
    if not (1e-8 < float(rv12) < 0.1):
        print(f"FAIL: rv_12bar out of range: {rv12}")
        return 5

    print("OK: inference outputs are in expected range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

