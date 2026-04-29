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
      per_element = alpha * Huber(log_rv_pred, log_rv_true)
                  + (1 - alpha) * QLIKE(rv_pred, rv_true)         # shape [B, H]
      per_horizon = mean over batch                                # shape [H]
      total       = weighted sum over horizons                     # scalar

    Backward compatibility: when ``horizon_weights is None`` (default), the
    horizon mean is unweighted, which is mathematically equivalent to taking
    a single mean over all [B, H] elements -- i.e. the legacy behavior of the
    pre-step-2 loss with ``reduction='mean'``.

    Long-horizon plan, step 2: ``horizon_weights`` allows giving more weight to
    distant horizons (e.g. rv_48bar_fwd, rv_288bar_fwd) so that they are not
    dominated by the larger absolute losses on short horizons.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        eps: float = 1e-8,
        huber_beta: float = 0.05,
        horizon_weights: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.huber = nn.SmoothL1Loss(beta=float(huber_beta), reduction="none")

        if horizon_weights is None:
            self._horizon_weights: torch.Tensor | None = None
        else:
            w = torch.as_tensor(tuple(horizon_weights), dtype=torch.float32)
            if w.ndim != 1 or w.numel() == 0:
                raise ValueError(
                    "horizon_weights must be a 1D non-empty sequence of floats."
                )
            if torch.any(w < 0):
                raise ValueError("horizon_weights must be non-negative.")
            total = float(w.sum().item())
            if total <= 0.0:
                raise ValueError("horizon_weights must sum to a positive value.")
            # Normalize once at construction so downstream sums weight to 1.
            self._horizon_weights = w / total

    def forward(self, log_rv_pred: torch.Tensor, rv_true: torch.Tensor) -> torch.Tensor:
        rv_true_pos = torch.clamp(rv_true, min=self.eps)
        log_rv_true = torch.log(rv_true_pos)
        huber_per = self.huber(log_rv_pred, log_rv_true)

        rv_pred_pos = torch.clamp(torch.exp(log_rv_pred), min=self.eps)
        qlike_per = torch.log(rv_pred_pos) + rv_true_pos / rv_pred_pos

        per_element = self.alpha * huber_per + (1.0 - self.alpha) * qlike_per

        if per_element.ndim < 2:
            # Defensive: 1D input means a single horizon -- equal-mean is the
            # only sensible reduction and weights would be meaningless here.
            return per_element.mean()

        per_horizon = per_element.mean(dim=0)  # [H]

        if self._horizon_weights is None:
            return per_horizon.mean()

        if self._horizon_weights.shape[0] != per_horizon.shape[0]:
            raise ValueError(
                f"horizon_weights length ({self._horizon_weights.shape[0]}) does "
                f"not match prediction horizons ({per_horizon.shape[0]})."
            )
        weights = self._horizon_weights.to(
            device=per_horizon.device, dtype=per_horizon.dtype
        )
        return (per_horizon * weights).sum()


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
