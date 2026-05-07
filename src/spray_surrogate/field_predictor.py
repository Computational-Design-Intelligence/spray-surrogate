"""Stage-2 U-Net: density -> (volume fraction, u, v)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv3x3 + BatchNorm + GELU layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FieldPredictor(nn.Module):
    """U-Net mapping density to [f, u, v]. f sigmoid-bounded; u, v unbounded."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.out_channels = out_channels

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        enc_channels: list[int] = []
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            enc_channels.append(out_ch)
            ch = out_ch

        bottleneck_ch = base_channels * (2**depth)
        self.bottleneck = ConvBlock(ch, bottleneck_ch)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_channels[i]
            self.upconvs.append(nn.Conv2d(ch, skip_ch, 1))
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))
            ch = skip_ch

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for enc, pool in zip(self.encoders, self.pools, strict=False):
            h = enc(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for upconv, dec, skip in zip(
            self.upconvs, self.decoders, reversed(skips), strict=False
        ):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = upconv(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
        raw = self.out_conv(h)
        # Channel 0: volume fraction in [0, 1]; channels 1, 2: unbounded velocities.
        return torch.cat([torch.sigmoid(raw[:, :1]), raw[:, 1:]], dim=1)


def load_field_predictor(checkpoint_path: str | Path, device: str = "cpu") -> FieldPredictor:
    """Load a trained field predictor from a checkpoint produced by `train-field-predictor`."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    model = FieldPredictor(
        in_channels=cfg.get("in_channels", 1),
        out_channels=cfg.get("out_channels", 3),
        base_channels=cfg.get("base_channels", 32),
        depth=cfg.get("depth", 4),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


def field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: dict[int, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Per-channel weighted MSE for [f, u, v]."""
    if weights is None:
        weights = {0: 1.0, 1: 1.0, 2: 1.0}
    losses: dict[str, torch.Tensor] = {}
    total: torch.Tensor | float = 0.0
    names = ["f", "u", "v"]
    for c in range(3):
        w = weights.get(c, 1.0)
        ch_loss = F.mse_loss(pred[:, c], target[:, c])
        losses[names[c]] = ch_loss
        total = total + w * ch_loss
    losses["total"] = total / sum(weights.values())
    return losses
