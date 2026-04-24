"""
Regression losses for trajectory forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MSELossWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_pred, y_true)


class RVLogAwareLoss(nn.Module):
    """
    Log-aware RV loss:
      total = alpha * Huber(log_rv_pred, log_rv_true)
            + (1 - alpha) * QLIKE(rv_pred, rv_true)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        eps: float = 1e-8,
        huber_beta: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.huber = nn.SmoothL1Loss(beta=float(huber_beta))

    def forward(self, log_rv_pred: torch.Tensor, rv_true: torch.Tensor) -> torch.Tensor:
        rv_true_pos = torch.clamp(rv_true, min=self.eps)
        log_rv_true = torch.log(rv_true_pos)
        huber_loss = self.huber(log_rv_pred, log_rv_true)

        rv_pred_pos = torch.clamp(torch.exp(log_rv_pred), min=self.eps)
        qlike_loss = torch.mean(torch.log(rv_pred_pos) + rv_true_pos / rv_pred_pos)
        return self.alpha * huber_loss + (1.0 - self.alpha) * qlike_loss


class JointLoss(nn.Module):
    """
    Joint loss for trajectory + auxiliary RV task:
      loss = mse_trajectory + alpha * mse_rv
    """

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.mse = nn.MSELoss()

    def forward(
        self,
        traj_pred: torch.Tensor,
        traj_true: torch.Tensor,
        rv_pred: torch.Tensor,
        rv_true: torch.Tensor,
    ) -> torch.Tensor:
        traj_loss = self.mse(traj_pred, traj_true)
        rv_loss = self.mse(rv_pred, rv_true)
        return traj_loss + self.alpha * rv_loss
