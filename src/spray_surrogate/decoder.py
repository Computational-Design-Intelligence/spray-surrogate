"""Time-conditioned SIREN decoder: (z, t, x, y) -> rho_hat(x, y; t)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .layers import (
    FiLM,
    FourierTimeFeatures,
    MultiScaleFourierFeatures,
    ResidualSirenLayer,
    SirenLayer,
)


class ConditioningNetwork(nn.Module):
    """Merge (z, gamma(t)) into a single conditioning vector."""

    def __init__(self, latent_dim: int, time_feat_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_feat_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

    def forward(self, z: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, t_feat], dim=-1))


class TimeConditionedDecoder(nn.Module):
    """Maps (z, t) at any (x, y) to a non-negative scalar density."""

    def __init__(
        self,
        latent_dim: int = 48,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_fourier_per_scale: int = 64,
        spatial_sigmas: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        time_frequencies: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        cond_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.spatial_fourier = MultiScaleFourierFeatures(
            input_dim=2, n_per_scale=n_fourier_per_scale, sigmas=spatial_sigmas
        )
        ff_dim = self.spatial_fourier.output_dim
        self.time_fourier = FourierTimeFeatures(frequencies=time_frequencies)
        self.conditioning = ConditioningNetwork(
            latent_dim=latent_dim,
            time_feat_dim=self.time_fourier.output_dim,
            cond_dim=cond_dim,
        )

        self.input_layer = SirenLayer(ff_dim, hidden_dim, omega=30.0, is_first=True)
        self.layers = nn.ModuleList(
            [ResidualSirenLayer(hidden_dim, omega=1.0) for _ in range(n_layers - 1)]
        )
        self.film_layers = nn.ModuleList(
            [FiLM(cond_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        nn.init.zeros_(self.out[-2].weight)
        nn.init.zeros_(self.out[-2].bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward_coords(
        self, z: torch.Tensor, t: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate at arbitrary query points. coords: (B, N, 2) -> (B, N, 1)."""
        _, N, _ = coords.shape
        ff = self.spatial_fourier(coords)
        t_feat = self.time_fourier(t)
        cond = self.conditioning(z, t_feat)
        cond_exp = cond.unsqueeze(1).expand(-1, N, -1)

        h = self.input_layer(ff)
        h = self.norms[0](h)
        h = self.film_layers[0](h, cond_exp)
        h = self.dropout(h)
        for layer, film, norm in zip(
            self.layers, self.film_layers[1:], self.norms[1:], strict=False
        ):
            h = layer(h)
            h = norm(h)
            h = film(h, cond_exp)
            h = self.dropout(h)
        return self.out(h)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        grid_shape: tuple[int, int] = (128, 128),
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Evaluate on a regular grid: (B, dz), (B,) -> (B, 1, Nx, Ny)."""
        B = z.size(0)
        Nx, Ny = grid_shape
        device = z.device
        xs = torch.linspace(x_range[0], x_range[1], Nx, device=device)
        ys = torch.linspace(y_range[0], y_range[1], Ny, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        coords = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2).expand(B, -1, -1)
        out_flat = self.forward_coords(z, t, coords)
        return out_flat.reshape(B, 1, Nx, Ny)
