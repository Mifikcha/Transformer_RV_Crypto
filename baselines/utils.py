"""
Shared utilities for baselines:
- RV regression helpers (multi-horizon)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Columns to exclude from features (targets, leakage, aux)
EXCLUDED_COLUMNS = {
    "ts",
    "target_return",
    "target_class",
    "is_valid_target",
    "future_close_30min",
    "future_close_60min",
    "future_close_120min",
    "future_close_240min",
    "delta_log_30min",
    "delta_log_60min",
    "delta_log_120min",
    "delta_log_240min",
    "delta_log_1bar_fwd",
    "delta_log_3bar_fwd",
    "delta_log_12bar_fwd",
    "delta_log_48bar_fwd",
    "delta_log_288bar_fwd",
    "rv_3bar_fwd",
    "rv_12bar_fwd",
    "rv_48bar_fwd",
    "rv_288bar_fwd",
    "base_regression",
    "base_class",
    "trading_class_optimistic",
    "trading_class_base",
    "trading_class_pessimistic",
}

VALID_TARGET_COL = "is_valid_target"

RV_TARGET_COLS = ("rv_3bar_fwd", "rv_12bar_fwd", "rv_48bar_fwd", "rv_288bar_fwd")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_recommended_features_path() -> str | None:
    """Return path to recommended_features.csv if it exists, else None.

    Resolution order:
    1) feature_selection/output/<symbol_lower>/recommended_features.csv
    2) feature_selection/output/recommended_features.csv
    """
    root = _project_root()
    symbol_lower = os.environ.get("SYMBOL", "BTCUSDT").strip().lower() or "btcusdt"
    candidates = [
        root / "feature_selection" / "output" / symbol_lower / "recommended_features.csv",
        root / "feature_selection" / "output" / "recommended_features.csv",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def load_recommended_features(path: str) -> list[str]:
    feat_df = pd.read_csv(path)
    if "feature" not in feat_df.columns:
        raise ValueError(f"'feature' column is missing in {path}")
    features = (
        feat_df["feature"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not features:
        raise ValueError(f"No recommended features in {path}")
    return features


def resolve_recommended_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Keep only existing numeric columns from the recommended feature list."""
    out: list[str] = []
    for f in features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            out.append(f)
    if not out:
        raise ValueError("No overlap between recommended features and dataset numeric columns.")
    return out


def get_feature_columns_recommended_or_all(df: pd.DataFrame) -> list[str]:
    """Prefer recommended_features.csv; fall back to full numeric feature set."""
    p = get_recommended_features_path()
    if p is None:
        return get_feature_columns(df)
    feats = load_recommended_features(p)
    return resolve_recommended_features(df, feats)


def rv_to_log(rv: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert RV in natural scale to log scale, clipping to avoid -inf."""
    return np.log(np.clip(rv.astype(float), eps, None))


def log_to_rv(y_log: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert log-RV predictions back to natural scale RV.

    Notes
    -----
    We clip the log-values before exp() to avoid numerical overflow and
    degenerate near-zero predictions that explode QLIKE.
    """
    y = np.asarray(y_log, dtype=float)
    y = np.clip(y, -50.0, 50.0)
    return np.clip(np.exp(y), eps, None)


def clip_log_predictions_to_train(
    y_pred_log: np.ndarray,
    y_train_log: np.ndarray,
    *,
    n_sigma: float = 5.0,
    hard_clip: tuple[float, float] = (-50.0, 50.0),
) -> np.ndarray:
    """Clip log predictions using train distribution (per horizon).

    This prevents a small number of extreme log-predictions from producing
    exp() blow-ups (huge MSE) or near-zero RV (huge QLIKE).
    """
    yp = np.asarray(y_pred_log, dtype=float)
    yt = np.asarray(y_train_log, dtype=float)
    if yt.ndim == 1:
        mu = float(np.nanmean(yt))
        sd = float(np.nanstd(yt))
        lo = mu - float(n_sigma) * sd
        hi = mu + float(n_sigma) * sd
        lo = max(lo, float(hard_clip[0]))
        hi = min(hi, float(hard_clip[1]))
        return np.clip(yp, lo, hi)

    # 2D: [N, H]
    mu = np.nanmean(yt, axis=0)
    sd = np.nanstd(yt, axis=0)
    lo = mu - float(n_sigma) * sd
    hi = mu + float(n_sigma) * sd
    lo = np.maximum(lo, float(hard_clip[0]))
    hi = np.minimum(hi, float(hard_clip[1]))
    return np.clip(yp, lo, hi)


def qlike_loss_logspace(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """QLIKE loss in log-space up to an additive constant.

    Using loss = y_pred + exp(y_true - y_pred).
    This is equivalent (up to constants) to log(v_hat) + v/v_hat.
    """
    yt = y_true_log.astype(float)
    yp = y_pred_log.astype(float)
    return float(np.mean(yp + np.exp(yt - yp)))

def get_default_data_path() -> str:
    """Resolve default dataset path with backward-compatible fallbacks.

    Resolution order:
    1) env var ``DATA_PATH`` (if points to existing file)
    2) symbol-aware path derived from env var ``SYMBOL`` (default ``BTCUSDT``)
    3) legacy BTC-only path(s) for backward compatibility
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.environ.get("DATA_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    symbol_lower = os.environ.get("SYMBOL", "BTCUSDT").strip().lower() or "btcusdt"

    candidates = [
        # Symbol-aware (priority): target/{symbol}_5m_final_with_targets.csv
        os.path.join(base, "target", f"{symbol_lower}_5m_final_with_targets.csv"),
        os.path.join(
            base, "dataset", "get_data", "output", "_main", f"{symbol_lower}_5m_final_with_targets.csv"
        ),
        # Legacy BTC-only fallbacks (only useful when SYMBOL=BTCUSDT and old layout).
        os.path.join(base, "target", "btcusdt_5m_final_with_targets.csv"),
        os.path.join(base, "dataset", "get_data", "output", "_main", "btcusdt_5m_final_with_targets.csv"),
        os.path.join(base, "target", "form_target", "btcusdt_5m_final_with_targets.csv"),
        os.path.join(base, "2. Target", "form_target", "btcusdt_5m_final_with_targets.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Keep deterministic default if nothing exists yet (symbol-aware target path).
    return candidates[0]


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV and apply RV-appropriate filtering.

    Important
    ---------
    The project dataset may contain a trading-oriented validity flag ``is_valid_target``.
    For RV regression experiments this is the wrong filter (it couples volatility modeling
    to a trading label definition). For RV we instead keep rows where RV forward targets
    are present (drop NaNs on the RV target columns).
    """
    df = pd.read_csv(path)
    rv_cols_present = [c for c in RV_TARGET_COLS if c in df.columns]
    if rv_cols_present:
        df = df.dropna(subset=rv_cols_present).copy()
    elif VALID_TARGET_COL in df.columns:
        df = df.loc[df[VALID_TARGET_COL].astype(int) == 1].copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (exclude targets and leakage)."""
    out: list[str] = []
    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def get_regression_target_columns(
    df: pd.DataFrame,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
) -> list[str]:
    cols = [c for c in target_columns if c in df.columns]
    if len(cols) != len(target_columns):
        missing = [c for c in target_columns if c not in cols]
        raise ValueError(f"Missing target columns in dataset: {missing}")
    return cols


def walk_forward_split(
    df: pd.DataFrame, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward: for each fold, train on past chunks, test on next chunk.
    Returns list of (train_idx, test_idx) arrays.

    Leakage note (purge/embargo)
    ----------------------------
    Our targets ``rv_*bar_fwd`` are forward-looking: for horizon H, label at t uses future
    data up to t+H-1. If we split at ``train_end``, the last H-1 labels in the train fold
    overlap with the first H-1 timestamps of the test fold (through the label construction),
    which is a form of look-ahead leakage. We therefore apply an embargo of max horizon
    rows to the end of every train fold.
    """
    def _max_horizon_from_rv_targets(cols: tuple[str, ...]) -> int:
        hs: list[int] = []
        for c in cols:
            # expected form: rv_{H}bar_fwd
            if not c.startswith("rv_") or "bar" not in c:
                continue
            try:
                mid = c.split("_", 2)[1]  # "{H}bar"
                h = int(mid.replace("bar", ""))
                hs.append(h)
            except Exception:
                continue
        return int(max(hs)) if hs else 0

    EMBARGO = _max_horizon_from_rv_targets(RV_TARGET_COLS)  # 288 in current config
    n = len(df)
    if n < n_splits + 1:
        raise ValueError("Not enough rows for walk-forward splits")
    segment_size = n // (n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_end = (k + 1) * segment_size
        test_end = (k + 2) * segment_size if (k + 2) <= n_splits else n
        train_end_eff = max(0, int(train_end) - int(EMBARGO))
        train_idx = np.arange(0, train_end_eff)
        test_idx = np.arange(train_end, min(test_end, n))
        if len(test_idx) == 0:
            continue
        if len(train_idx) == 0:
            raise ValueError(
                f"Embargo={EMBARGO} leaves empty train fold at split {k}. "
                f"Increase data length or reduce n_splits."
            )
        splits.append((train_idx, test_idx))
    return splits


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> dict[str, float]:
    """
    Regression metrics per horizon and their macro averages.
    Expects shape [N, H] for y_true and y_pred.
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("compute_regression_metrics expects 2D arrays [N, H].")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.shape[1] != len(target_columns):
        raise ValueError(
            f"target_columns length ({len(target_columns)}) "
            f"must match y shape second dim ({y_true.shape[1]})."
        )

    out: dict[str, float] = {}
    mse_list: list[float] = []
    mae_list: list[float] = []
    r2_list: list[float] = []
    da_list: list[float] = []
    qlike_list: list[float] = []
    hmse_list: list[float] = []

    for idx, col in enumerate(target_columns):
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]
        mse_v = float(mean_squared_error(y_t, y_p))
        mae_v = float(mean_absolute_error(y_t, y_p))
        eps = 1e-12
        yt_pos = np.clip(y_t, eps, None)
        yp_pos = np.clip(y_p, eps, None)
        # Important: for log-space models, R² in natural space after exp()
        # becomes hard to interpret due to exponential error amplification.
        # We therefore compute R² in log-space for baselines.
        r2_v = float(r2_score(np.log(yt_pos), np.log(yp_pos)))
        # Directional Accuracy for RV should measure direction of *changes*,
        # not sign(RV) (RV is positive -> degenerate DA==1).
        if len(yt_pos) >= 2:
            dyt = np.diff(np.log(yt_pos.astype(float)))
            dyp = np.diff(np.log(yp_pos.astype(float)))
            da_v = float(np.mean(np.sign(dyt) == np.sign(dyp)))
        else:
            da_v = float("nan")
        qlike_v = float(np.mean(np.log(yp_pos) + yt_pos / yp_pos))
        hmse_v = float(np.mean(((y_p.astype(float) - yt_pos) / yt_pos) ** 2))
        out[f"mse_{col}"] = mse_v
        out[f"mae_{col}"] = mae_v
        out[f"r2_{col}"] = r2_v
        out[f"da_{col}"] = da_v
        out[f"qlike_{col}"] = qlike_v
        out[f"hmse_{col}"] = hmse_v
        mse_list.append(mse_v)
        mae_list.append(mae_v)
        r2_list.append(r2_v)
        da_list.append(da_v)
        qlike_list.append(qlike_v)
        hmse_list.append(hmse_v)

    out["mse_mean"] = float(np.mean(mse_list))
    out["mae_mean"] = float(np.mean(mae_list))
    out["r2_mean"] = float(np.mean(r2_list))
    out["da_mean"] = float(np.mean(da_list)) if da_list else float("nan")
    out["qlike_mean"] = float(np.mean(qlike_list))
    out["hmse_mean"] = float(np.mean(hmse_list))
    return out


def print_regression_metrics(
    metrics_per_fold: list[dict[str, float]],
    model_name: str,
    target_columns: list[str],
) -> None:
    """Print regression metrics aggregated over folds (mean +- std)."""
    if not metrics_per_fold:
        print(f"[{model_name}] No folds.")
        return

    print("\n" + "=" * 60)
    print(f"  {model_name}")
    print("=" * 60)
    for metric_name in ("mse", "mae", "r2", "hmse"):
        mean_key = f"{metric_name}_mean"
        vals = [float(m[mean_key]) for m in metrics_per_fold if mean_key in m]
        if vals:
            print(
                f"  {metric_name.upper():<4} mean: {np.mean(vals):.6f} "
                f"(+- {np.std(vals):.6f})"
            )
    print("  --- Per-horizon (mean over folds) ---")
    for col in target_columns:
        mse_vals = [float(m[f"mse_{col}"]) for m in metrics_per_fold if f"mse_{col}" in m]
        mae_vals = [float(m[f"mae_{col}"]) for m in metrics_per_fold if f"mae_{col}" in m]
        r2_vals = [float(m[f"r2_{col}"]) for m in metrics_per_fold if f"r2_{col}" in m]
        hmse_vals = [float(m[f"hmse_{col}"]) for m in metrics_per_fold if f"hmse_{col}" in m]
        if not mse_vals:
            continue
        print(
            f"    {col}: MSE {np.mean(mse_vals):.6f}, MAE {np.mean(mae_vals):.6f}, "
            f"R2 {np.mean(r2_vals):.4f}, "
            f"HMSE {np.mean(hmse_vals):.6f}, "
            f"QLIKE {np.mean([float(m[f'qlike_{col}']) for m in metrics_per_fold if f'qlike_{col}' in m]):.6f}"
        )
    print("=" * 60 + "\n")
