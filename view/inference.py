"""
Inference wrapper: load a trained .pt checkpoint and run predictions.

The checkpoint payload (saved by transformer/train.py) contains:
  state_dict, model_config, effective_n_horizons, train_config,
  feature_columns, target_columns, scaler_mean, scaler_scale,
  har_scaler_mean, har_scaler_scale, n_har_context, fold_id, run_name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from transformer.model import build_model

logger = logging.getLogger(__name__)


def _restore_scaler(mean: list[float], scale: list[float]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(mean, dtype=np.float64)
    scaler.scale_ = np.array(scale, dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(mean)
    scaler.n_samples_seen_ = 1
    return scaler


class RVInference:
    @dataclass
    class _ModelPack:
        model: torch.nn.Module
        feature_scaler: StandardScaler
        har_scaler: StandardScaler
        target_columns: list[str]
        model_ver: str
        fold_id: int | str
        path: str

    def __init__(self, model_paths: str | list[str], features_path: str | None = None) -> None:
        if isinstance(model_paths, str):
            paths = [x.strip() for x in model_paths.split(",") if x.strip()]
        else:
            paths = [x.strip() for x in model_paths if x.strip()]
        if not paths:
            raise ValueError("RVInference requires at least one model path")

        self.models: list[RVInference._ModelPack] = []
        self.feature_columns: list[str] = []
        self.target_columns: list[str] = []
        self.har_bounds: list[tuple[float, float]] = []
        self.log_bias = {"rv_3bar": 0.0, "rv_12bar": 0.0}
        self.log_bias_sigma = {"rv_3bar": 0.0, "rv_12bar": 0.0}

        for idx, model_path in enumerate(paths):
            logger.info("Loading checkpoint from %s", model_path)
            payload = torch.load(model_path, map_location="cpu", weights_only=False)

            feature_columns: list[str] = payload["feature_columns"]
            target_columns: list[str] = payload["target_columns"]
            feature_scaler = _restore_scaler(payload["scaler_mean"], payload["scaler_scale"])
            har_scaler = _restore_scaler(payload["har_scaler_mean"], payload["har_scaler_scale"])

            if idx == 0:
                self.feature_columns = feature_columns
                self.target_columns = target_columns
                mean = np.array(payload["har_scaler_mean"], dtype=np.float64)
                scale = np.array(payload["har_scaler_scale"], dtype=np.float64)
                self.har_bounds = [
                    (float(m - 5.0 * s), float(m + 5.0 * s))
                    for m, s in zip(mean, scale)
                ]
            else:
                if feature_columns != self.feature_columns:
                    raise ValueError("All ensemble models must share identical feature_columns")
                if target_columns != self.target_columns:
                    raise ValueError("All ensemble models must share identical target_columns")

            mcfg = payload["model_config"]
            n_horizons = payload.get("effective_n_horizons", len(target_columns))
            n_har = payload.get("n_har_context", 6)

            model = build_model(
                model_type=mcfg["model_type"],
                input_dim=len(feature_columns),
                seq_len=mcfg["seq_len"],
                patch_size=mcfg["patch_size"],
                d_model=mcfg["d_model"],
                n_heads=mcfg["n_heads"],
                n_layers=mcfg["n_layers"],
                d_ff=mcfg["d_ff"],
                dropout=mcfg["dropout"],
                n_horizons=n_horizons,
                n_enc_layers=mcfg.get("n_enc_layers", mcfg["n_layers"]),
                n_dec_layers=mcfg.get("n_dec_layers", mcfg["n_layers"]),
                n_har=n_har,
                d_har=16,
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            fold_id = payload.get("fold_id", "?")
            model_ver = f"fold_{fold_id}_{Path(model_path).stem}"
            self.models.append(
                RVInference._ModelPack(
                    model=model,
                    feature_scaler=feature_scaler,
                    har_scaler=har_scaler,
                    target_columns=target_columns,
                    model_ver=model_ver,
                    fold_id=fold_id,
                    path=model_path,
                )
            )
            logger.info(
                "Model loaded: %s from %s, features=%d, horizons=%d",
                mcfg["model_type"], model_path, len(feature_columns), n_horizons,
            )

        if len(self.models) == 1:
            self.model_ver = self.models[0].model_ver
        else:
            folds = ",".join(str(m.fold_id) for m in self.models)
            self.model_ver = f"ensemble_{len(self.models)}f[{folds}]"

    def update_log_bias(
        self,
        lb3: float | None = None,
        lb12: float | None = None,
        ls3: float | None = None,
        ls12: float | None = None,
    ) -> None:
        if lb3 is not None:
            self.log_bias["rv_3bar"] = float(lb3)
        if lb12 is not None:
            self.log_bias["rv_12bar"] = float(lb12)
        if ls3 is not None:
            self.log_bias_sigma["rv_3bar"] = float(ls3)
        if ls12 is not None:
            self.log_bias_sigma["rv_12bar"] = float(ls12)

    def predict(self, window: np.ndarray, har_context: np.ndarray) -> dict[str, float]:
        """
        Run inference on a raw (un-scaled) feature window.

        Args:
            window: [seq_len, n_features] raw features.
            har_context: [n_har] raw HAR context scalars.

        Returns:
            {"rv_3bar": float, "rv_12bar": float}  (natural scale, after exp).
        """
        preds: list[np.ndarray] = []
        for pack in self.models:
            window_scaled = pack.feature_scaler.transform(window).astype(np.float32)
            har_scaled = pack.har_scaler.transform(har_context[np.newaxis]).astype(np.float32)
            x = torch.from_numpy(window_scaled).unsqueeze(0)   # [1, seq_len, F]
            har = torch.from_numpy(har_scaled)                 # [1, n_har]
            with torch.no_grad():
                out = pack.model(x, har)
            pred = torch.exp(out[0]).clamp(min=1e-8).numpy()
            preds.append(pred)

        pred_mean = np.mean(np.stack(preds, axis=0), axis=0)
        result: dict[str, float] = {}
        for i, col in enumerate(self.target_columns):
            key = col.replace("_fwd", "")
            val = float(pred_mean[i])
            # Step 2 (Jensen / multiplicative bias calibration in log domain)
            if key in self.log_bias:
                val *= float(np.exp(self.log_bias[key]))
            result[key] = val
        return result
