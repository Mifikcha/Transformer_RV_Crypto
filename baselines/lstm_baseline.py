"""LSTM baseline for multi-horizon RV forecasting."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils import (
    RV_TARGET_COLS,
    compute_regression_metrics,
    get_default_data_path,
    get_feature_columns_recommended_or_all,
    get_regression_target_columns,
    log_to_rv,
    load_dataset,
    print_regression_metrics,
    rv_to_log,
    walk_forward_split,
)


class _SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, seq_len: int) -> None:
        if len(x) != len(y):
            raise ValueError("x/y length mismatch")
        if len(x) < seq_len:
            raise ValueError("Not enough rows for one sequence")
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.seq_len = int(seq_len)
        self.end_idx = np.arange(seq_len - 1, len(x), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.end_idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.end_idx[idx])
        start = end - self.seq_len + 1
        return torch.from_numpy(self.x[start : end + 1]), torch.from_numpy(self.y[end])


class _LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, out_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
    seq_len: int = 240,
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 128,
    max_epochs: int = 75,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    patience: int = 15,
    warmup_epochs: int = 5,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns_recommended_or_all(df)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)
    X = df[feat_cols].astype(float).fillna(0.0).values
    y = df[tgt_cols].astype(float).values
    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _warmup_cosine(epoch: int) -> float:
        w = int(warmup_epochs)
        if epoch < max(w, 1):
            return float(lr) * (epoch + 1) / max(w, 1)
        denom = max(int(max_epochs) - w, 1)
        progress = (epoch - w) / denom
        return float(lr) * 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    def _qlike_loss_log(pred_log: torch.Tensor, true_log: torch.Tensor) -> torch.Tensor:
        # L = y_pred + exp(y_true - y_pred)
        return (pred_log + torch.exp(true_log - pred_log)).mean()

    for train_idx, test_idx in splits:
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)

        y_train_log = rv_to_log(y_train).astype(np.float32)
        y_test_log = rv_to_log(y_test).astype(np.float32)

        train_ds = _SeqDataset(X_train, y_train_log, seq_len=seq_len)
        ctx = seq_len - 1
        if ctx > 0:
            X_test_full = np.concatenate([X_train[-ctx:], X_test], axis=0)
            y_test_full = np.concatenate([y_train_log[-ctx:], y_test_log], axis=0)
        else:
            X_test_full = X_test
            y_test_full = y_test_log
        test_ds = _SeqDataset(X_test_full, y_test_full, seq_len=seq_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        model = _LSTMRegressor(
            input_dim=X.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=len(tgt_cols),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))

        best_state = None
        best_val = float("inf")
        wait = 0
        for epoch in range(max_epochs):
            # Match transformer-style schedule roughly.
            lr_eff = _warmup_cosine(epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_eff
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = _qlike_loss_log(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    val_losses.append(float(_qlike_loss_log(model(xb), yb).detach().cpu().item()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        pred_parts: list[np.ndarray] = []
        true_parts: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred_parts.append(model(xb).detach().cpu().numpy())
                true_parts.append(yb.detach().cpu().numpy())
        y_pred_log = np.concatenate(pred_parts, axis=0)
        y_true_log = np.concatenate(true_parts, axis=0)
        y_pred = log_to_rv(y_pred_log)
        y_true = log_to_rv(y_true_log)
        metrics_per_fold.append(
            compute_regression_metrics(y_true=y_true, y_pred=y_pred, target_columns=tgt_cols)
        )

    print_regression_metrics(metrics_per_fold, "LSTM (2-layer, hidden=64, log-target, QLIKE)", tgt_cols)
    return metrics_per_fold


if __name__ == "__main__":
    run()
