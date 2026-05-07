from __future__ import annotations

import sys
import argparse
import json

import numpy as np
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.garch_models._common import (
    data_dir,
    get_symbol_lower,
    metrics_for_variance,
    output_dir,
)
from baselines.garch_models._fit_predict import fit_forecast_variance


MODEL_NAME = "EGARCH(1,1)"


def run(*, symbol_lower: str, n_splits: int = 5) -> dict:
    inp = data_dir(symbol_lower) / "daily_variance.parquet"
    if not inp.exists():
        raise FileNotFoundError(
            f"Daily variance dataset not found: {inp}. "
            "Run prepare_daily_variance_target.py first."
        )

    df = pd.read_parquet(inp)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    r = df["ret_1d"].to_numpy(dtype=float)
    y = df["var_1d"].to_numpy(dtype=float)

    folds, y_pred_all = fit_forecast_variance(
        returns=r,
        realized_var=y,
        n_splits=int(n_splits),
        mean="Zero",
        vol="EGARCH",
        p=1,
        q=1,
        o=0,
        dist="normal",
    )

    mask = np.isfinite(y_pred_all)
    overall = metrics_for_variance(y_true=y[mask], y_pred=y_pred_all[mask]) if np.any(mask) else {}
    r2_oos_vals = [
        float(fr.metrics.get("r2_log_oos"))
        for fr in folds
        if isinstance(fr.metrics.get("r2_log_oos"), (int, float)) and np.isfinite(fr.metrics.get("r2_log_oos"))
    ]
    if overall and r2_oos_vals:
        overall["r2_log_oos_mean"] = float(np.mean(r2_oos_vals))

    out = output_dir(symbol_lower)
    pred_path = out / "predictions_egarch11.parquet"
    metrics_path = out / "metrics_egarch11.json"

    pred_df = df[["ts", "ret_1d", "var_1d"]].copy()
    pred_df["var_pred"] = y_pred_all
    pred_df.to_parquet(pred_path, index=False)

    payload = {
        "model": MODEL_NAME,
        "symbol_lower": symbol_lower,
        "n_splits": int(n_splits),
        "folds": [f.__dict__ for f in folds],
        "overall": overall,
        "artifacts": {"predictions": str(pred_path), "metrics": str(metrics_path)},
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol-lower", type=str, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    symbol_lower = (args.symbol_lower or get_symbol_lower()).strip().lower()
    res = run(symbol_lower=symbol_lower, n_splits=int(args.n_splits))
    print(json.dumps(res["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

