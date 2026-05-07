"""Shared neural-network building blocks: SIREN, FiLM, Fourier features."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    """y = sin(omega * (W x + b)) with the Sitzmann et al. initialisation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 1.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class ResidualSirenLayer(nn.Module):
    """SIREN with a residual connection."""

    def __init__(self, dim: int, omega: float = 1.0) -> None:
        super().__init__()
        self.siren = SirenLayer(dim, dim, omega=omega, is_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.siren(x)


class FiLM(nn.Module):
    """Feature-wise linear modulation: h <- (1 + gamma(c)) * h + beta(c)."""

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(c).chunk(2, dim=-1)
        return (1.0 + scale) * h + shift


class MultiScaleFourierFeatures(nn.Module):
    """Random Fourier features at multiple bandwidths."""

    def __init__(
        self,
        input_dim: int = 2,
        n_per_scale: int = 64,
        sigmas: Sequence[float] = (1.0, 5.0, 15.0, 50.0),
    ) -> None:
        super().__init__()
        self.n_scales = len(sigmas)
        self.output_dim = 2 * n_per_scale * len(sigmas)
        for i, sigma in enumerate(sigmas):
            B = torch.randn(input_dim, n_per_scale) * sigma
            self.register_buffer(f"B_{i}", B)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i in range(self.n_scales):
            proj = coords @ getattr(self, f"B_{i}")
            parts.extend([proj.sin(), proj.cos()])
        return torch.cat(parts, dim=-1)


class FourierTimeFeatures(nn.Module):
    """Sinusoidal time features at fixed frequencies, plus the raw value t."""

    def __init__(
        self,
        frequencies: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0),
    ) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.tensor(list(frequencies), dtype=torch.float32))
        self.output_dim = 2 * len(frequencies) + 1

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        phase = t * self.freqs
        return torch.cat([phase.sin(), phase.cos(), t], dim=-1)
