"""
Transformer model variants for RV forecasting.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _build_causal_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.ones(length, length), diagonal=1).bool()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = x.size(1)
        return x + self.pe[:, :l, :]


class EncoderBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.encoder(x, mask=mask)
        return self.norm(x)

class EncoderOnlyRegressor(nn.Module):
    """Model A: patch encoder-only regressor with HAR global context (v4 step 1+3: per-horizon heads)."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        patch_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        n_horizons: int,
        n_har: int = 6,
        d_har: int = 16,
    ) -> None:
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size")
        self.n_horizons = int(n_horizons)
        self.n_har = int(n_har)
        self.d_har = int(d_har)
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_dim = patch_size * input_dim
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=self.n_patches)
        self.backbone = EncoderBackbone(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        if self.n_har > 0:
            self.har_proj = nn.Sequential(
                nn.Linear(self.n_har, self.d_har),
                nn.GELU(),
            )
            head_in = d_model + self.d_har
        else:
            self.har_proj = None
            head_in = d_model
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_in, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )
                for _ in range(self.n_horizons)
            ]
        )

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f = x.shape
        x = x.view(b, self.n_patches, self.patch_size, f)
        return x.reshape(b, self.n_patches, self.patch_dim)

    def forward(self, x: torch.Tensor, har: torch.Tensor | None = None) -> torch.Tensor:
        x = self._to_patches(x)
        x = self.patch_proj(x)
        x = self.pos_enc(x)
        x = self.backbone(x)
        pooled = x.mean(dim=1)
        if self.har_proj is None:
            combined = pooled
        else:
            if har is None:
                raise ValueError("HAR tensor is required when n_har > 0.")
            har_emb = self.har_proj(har)
            combined = torch.cat([pooled, har_emb], dim=-1)
        return torch.cat([h(combined) for h in self.heads], dim=-1)


class DecoderOnlyTransformer(nn.Module):
    """Model B: decoder-only with causal mask."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        n_horizons: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=seq_len)
        self.backbone = EncoderBackbone(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons),
        )
        self.register_buffer("causal_mask", _build_causal_mask(seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.backbone(x, mask=self.causal_mask)
        x = self.final_norm(x[:, -1, :])
        return self.head(x)


class PatchDecoderTransformer(nn.Module):
    """Model D: patching + decoder-only causal behavior."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        patch_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        n_horizons: int,
    ) -> None:
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size")
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_dim = patch_size * input_dim
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=self.n_patches)
        self.backbone = EncoderBackbone(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons),
        )
        self.register_buffer(
            "causal_mask", _build_causal_mask(self.n_patches), persistent=False
        )

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f = x.shape
        x = x.view(b, self.n_patches, self.patch_size, f)
        return x.reshape(b, self.n_patches, self.patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_patches(x)
        x = self.patch_proj(x)
        x = self.pos_enc(x)
        x = self.backbone(x, mask=self.causal_mask)
        x = self.final_norm(x[:, -1, :])
        return self.head(x)


class VanillaTransformerRegressor(nn.Module):
    """Model C: encoder-decoder transformer regressor."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_enc_layers: int,
        n_dec_layers: int,
        d_ff: int,
        dropout: float,
        n_horizons: int,
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.enc_proj = nn.Linear(input_dim, d_model)
        self.enc_pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        self.dec_input = nn.Parameter(torch.randn(1, n_horizons, d_model) * 0.02)
        self.dec_pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=n_horizons)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        self.register_buffer("tgt_mask", _build_causal_mask(n_horizons), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        memory = self.enc_proj(x)
        memory = self.enc_pe(memory)
        memory = self.encoder(memory)
        tgt = self.dec_input.expand(b, -1, -1)
        tgt = self.dec_pe(tgt)
        out = self.decoder(tgt, memory, tgt_mask=self.tgt_mask)
        out = self.final_norm(out)
        return self.out_proj(out).squeeze(-1)


def build_model(
    model_type: str,
    input_dim: int,
    seq_len: int,
    patch_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    n_horizons: int = 5,
    n_enc_layers: int | None = None,
    n_dec_layers: int | None = None,
    n_har: int = 6,
    d_har: int = 16,
) -> nn.Module:
    model_type = model_type.lower().strip()
    n_enc = n_layers if n_enc_layers is None else n_enc_layers
    n_dec = n_layers if n_dec_layers is None else n_dec_layers

    if model_type in {"patch_encoder", "encoder_only"}:
        return EncoderOnlyRegressor(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc,
            d_ff=d_ff,
            dropout=dropout,
            n_horizons=n_horizons,
            n_har=n_har,
            d_har=d_har,
        )
    if model_type == "decoder_only":
        return DecoderOnlyTransformer(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec,
            d_ff=d_ff,
            dropout=dropout,
            n_horizons=n_horizons,
        )
    if model_type == "vanilla_enc_dec":
        return VanillaTransformerRegressor(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_enc_layers=n_enc,
            n_dec_layers=n_dec,
            d_ff=d_ff,
            dropout=dropout,
            n_horizons=n_horizons,
        )
    if model_type == "patch_decoder":
        return PatchDecoderTransformer(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec,
            d_ff=d_ff,
            dropout=dropout,
            n_horizons=n_horizons,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
