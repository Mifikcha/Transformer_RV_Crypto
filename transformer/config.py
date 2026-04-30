"""
Configuration for RV forecasting pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _symbol_lower() -> str:
    """Resolve the active trading symbol (lowercase) from the SYMBOL env var.

    Defaults to ``btcusdt`` so existing BTC-only workflows keep working.
    """
    return os.environ.get("SYMBOL", "BTCUSDT").strip().lower() or "btcusdt"


SYMBOL_LOWER = _symbol_lower()

DEFAULT_DATA_PATH = os.path.join(
    PROJECT_ROOT, "target", f"{SYMBOL_LOWER}_5m_final_with_targets.csv"
)
DEFAULT_FEATURES_DIR = os.path.join(PROJECT_ROOT, "feature_selection", "output", SYMBOL_LOWER)
DEFAULT_FEATURES_PATH = os.path.join(
    DEFAULT_FEATURES_DIR, "recommended_features.csv"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "transformer", "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models", SYMBOL_LOWER)
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions", SYMBOL_LOWER)
PICTURES_DIR = os.path.join(OUTPUT_DIR, "pictures", SYMBOL_LOWER)
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs", SYMBOL_LOWER)


@dataclass(frozen=True)
class ModelConfig:
    # architecture
    model_type: str = "patch_encoder"  # patch_encoder|decoder_only|vanilla_enc_dec|patch_decoder
    seq_len: int = 240
    patch_size: int = 12
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3  # legacy/shared
    n_enc_layers: int = 3
    n_dec_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.2
    n_horizons: int = 5
    har_mode: str = "full"  # full|none|weekly_only


@dataclass(frozen=True)
class TrainConfig:
    # task
    task: str = "rv"
    # long-horizon plan step 1: train on all 4 forward RV horizons.
    target_columns: tuple[str, ...] = (
        "rv_3bar_fwd",
        "rv_12bar_fwd",
        "rv_48bar_fwd",
        "rv_288bar_fwd",
    )

    # optimization (plan v4 step 2: conservative training + warmup)
    batch_size: int = 128
    lr: float = 5e-5
    max_epochs: int = 75
    patience: int = 15
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    loss_type: str = "rv_log_aware"  # rv_log_aware|mse
    # long-horizon plan step 1: switch to pure QLIKE per ablation study (B1.4).
    loss_alpha: float = 0.0
    # long-horizon plan step 2: per-horizon weights for the RV loss.
    # Empty tuple = equal weighting (legacy / backward-compatible behavior).
    # When non-empty, length must match len(target_columns) and the values are
    # normalized to sum to 1 inside RVLogAwareLoss.
    horizon_weights: tuple[float, ...] = ()

    # cv and reproducibility
    n_splits: int = 5
    seed: int = 42
    num_workers: int = 0

@dataclass(frozen=True)
class AppConfig:
    data_path: str = DEFAULT_DATA_PATH
    features_path: str = DEFAULT_FEATURES_PATH
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, MODELS_DIR, PREDICTIONS_DIR, PICTURES_DIR, LOGS_DIR):
        os.makedirs(path, exist_ok=True)
