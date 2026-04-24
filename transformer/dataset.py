"""
Dataset helpers for sequence modeling with walk-forward split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from baselines.utils import (
    RV_TARGET_COLS,
    get_feature_columns,
    load_dataset,
    walk_forward_split,
)

# HAR context (plan v6 step 1): use less-noisy GK-based RV as global scalars
HAR_RV_BASE_COLS = ("rv_gk_15min", "rv_gk_60min", "rv_gk_240min")
WEEK_BARS = 5 * 288
MONTH_BARS = 22 * 288
# weekly + monthly per base column
N_HAR_CONTEXT = len(HAR_RV_BASE_COLS) * 2


def add_rv_har_context_columns(df: pd.DataFrame, mode: str = "full") -> list[str]:
    """
    Adds HAR context columns in-place. Rolling is causal (past only).
    mode:
      - full: weekly + monthly HAR columns
      - weekly_only: weekly HAR columns only
      - none: no HAR columns
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"full", "none", "weekly_only"}:
        raise ValueError(f"Unsupported har mode: {mode!r}")
    if mode_norm == "none":
        return []

    har_cols: list[str] = []
    for col in HAR_RV_BASE_COLS:
        if col not in df.columns:
            raise ValueError(
                f"HAR base column '{col}' missing from dataframe (required for HAR context)."
            )
        w_name = f"{col}_har_w"
        m_name = f"{col}_har_m"
        df[w_name] = df[col].rolling(WEEK_BARS, min_periods=WEEK_BARS // 2).mean()
        har_cols.append(w_name)
        if mode_norm == "full":
            df[m_name] = df[col].rolling(MONTH_BARS, min_periods=MONTH_BARS // 2).mean()
            har_cols.append(m_name)
    df[har_cols] = df[har_cols].bfill().ffill()
    return har_cols


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


def load_base_dataframe(data_path: str) -> pd.DataFrame:
    return load_dataset(data_path)


def resolve_features(df: pd.DataFrame, recommended_features: list[str]) -> list[str]:
    numeric_features = set(get_feature_columns(df))
    selected = [f for f in recommended_features if f in numeric_features]
    if not selected:
        raise ValueError("No overlap between recommended features and dataset columns.")
    return selected


@dataclass(frozen=True)
class RegressionFoldData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray  # [N, H]
    ts_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray  # [N, H]
    ts_val: np.ndarray
    scaler: StandardScaler
    har_train: np.ndarray  # [N_train, n_har], StandardScaler applied (train fit)
    har_val: np.ndarray  # [N_val, n_har]
    har_scaler: StandardScaler
    fold_id: int


class TimeSeriesRegressionDataset(Dataset):
    """
    Returns windows:
      x = features[start:end+1]  -> [seq_len, F]
      har = har_values[end]      -> [n_har]  (global context at last bar of window)
      y = targets[end]           -> [H]
    """

    def __init__(
        self,
        features_array: np.ndarray,
        targets_array: np.ndarray,
        seq_len: int,
        har_array: np.ndarray | None = None,
    ) -> None:
        if seq_len <= 1:
            raise ValueError("seq_len must be > 1")
        if len(features_array) != len(targets_array):
            raise ValueError("features and targets length mismatch")
        if len(features_array) < seq_len:
            raise ValueError("Not enough rows for one sequence.")
        if targets_array.ndim != 2:
            raise ValueError("targets_array must be 2D [N, H]")
        if har_array is not None and len(har_array) != len(features_array):
            raise ValueError("har_array length must match features_array.")

        self.features = features_array.astype(np.float32, copy=False)
        self.targets = targets_array.astype(np.float32, copy=False)
        self.har_values = (
            None
            if har_array is None
            else har_array.astype(np.float32, copy=False)
        )
        self.seq_len = seq_len
        self.end_indices = np.arange(seq_len - 1, len(self.features), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end_idx = int(self.end_indices[idx])
        start_idx = end_idx - self.seq_len + 1
        x = self.features[start_idx : end_idx + 1]
        y = self.targets[end_idx]
        out: dict[str, torch.Tensor] = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "end_idx": torch.tensor(end_idx, dtype=torch.long),
        }
        if self.har_values is not None:
            out["har"] = torch.from_numpy(self.har_values[end_idx])
        return out


def make_fold_data_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str] | None,
    n_splits: int,
    har_mode: str = "full",
) -> list[RegressionFoldData]:
    splits = walk_forward_split(df, n_splits=n_splits)
    if target_cols is None:
        target_cols = [c for c in RV_TARGET_COLS if c in df.columns]
    if not target_cols:
        raise ValueError("No regression target columns provided/found.")

    har_cols = add_rv_har_context_columns(df, mode=har_mode)

    y_all = df[target_cols].astype(float).values  # [N, H]
    ts_all = df["ts"].astype(str).values
    X_all = df[feature_cols].astype(float).fillna(0.0).values
    h_dim = len(har_cols)
    if h_dim > 0:
        H_all = df[har_cols].astype(float).values
    else:
        H_all = np.zeros((len(df), 0), dtype=np.float32)

    fold_data: list[RegressionFoldData] = []
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        X_train_raw = X_all[train_idx]
        X_val_raw = X_all[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val = scaler.transform(X_val_raw).astype(np.float32)

        H_train_raw = H_all[train_idx]
        H_val_raw = H_all[val_idx]
        if h_dim > 0:
            har_scaler = StandardScaler()
            har_train = har_scaler.fit_transform(H_train_raw).astype(np.float32)
            har_val = har_scaler.transform(H_val_raw).astype(np.float32)
        else:
            har_scaler = StandardScaler()
            har_scaler.fit(np.zeros((1, 1), dtype=np.float32))
            har_train = np.zeros((len(train_idx), 0), dtype=np.float32)
            har_val = np.zeros((len(val_idx), 0), dtype=np.float32)

        fold_data.append(
            RegressionFoldData(
                train_idx=train_idx,
                val_idx=val_idx,
                X_train=X_train,
                y_train=y_all[train_idx].astype(np.float32),
                ts_train=ts_all[train_idx],
                X_val=X_val,
                y_val=y_all[val_idx].astype(np.float32),
                ts_val=ts_all[val_idx],
                scaler=scaler,
                har_train=har_train,
                har_val=har_val,
                har_scaler=har_scaler,
                fold_id=fold_id,
            )
        )
    return fold_data
