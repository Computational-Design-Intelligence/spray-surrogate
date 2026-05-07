"""Stage-1 VAE: FNO encoder + time-conditioned decoder + 1D geometry head."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import TimeConditionedDecoder
from .encoder import FNOEncoder
from .geometry_head import GeometryHead


class FNOVAE(nn.Module):
    """rho -> z -> (rho_hat(t), nozzle geometry)."""

    def __init__(
        self,
        latent_dim: int = 48,
        d_model: int = 64,
        n_fno_blocks: int = 4,
        n_modes: int = 16,
        decoder_hidden: int = 128,
        decoder_layers: int = 4,
        decoder_fourier: int = 64,
        spatial_sigmas: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        time_frequencies: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        cond_dim: int = 128,
        geom_hidden: int = 64,
        geom_layers: int = 3,
        geom_fourier: int = 32,
        x_lo_nozzle: float = 0.0,
        x_hi_nozzle: float = 0.7,
        geom_dim: int = 200,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.geom_dim = geom_dim
        M = geom_dim // 2

        self.encoder = FNOEncoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=n_fno_blocks,
            n_modes_x=n_modes,
            n_modes_y=n_modes,
        )
        self.decoder = TimeConditionedDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden,
            n_layers=decoder_layers,
            n_fourier_per_scale=decoder_fourier,
            spatial_sigmas=spatial_sigmas,
            time_frequencies=time_frequencies,
            cond_dim=cond_dim,
            dropout=dropout,
        )
        self.geom_head = GeometryHead(
            latent_dim=latent_dim,
            hidden_dim=geom_hidden,
            n_layers=geom_layers,
            n_fourier=geom_fourier,
            x_lo=x_lo_nozzle,
            x_hi=x_hi_nozzle,
        )
        self.register_buffer(
            "x_canonical", torch.linspace(x_lo_nozzle, x_hi_nozzle, M)
        )

    def forward(
        self, rho: torch.Tensor, t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = rho.shape[0]
        Nx, Ny = rho.shape[-2:]
        if t is None:
            t = torch.zeros(B, device=rho.device)
        mu, logvar = self.encoder(rho)
        z = self.encoder.reparameterize(mu, logvar, self.training)
        rho_hat = self.decoder(z, t, grid_shape=(Nx, Ny))
        # Use mu (not z) for geometry to give a noise-free curve target.
        geom_hat = self.geom_head.evaluate_at_canonical(mu, self.x_canonical)
        return rho_hat, mu, logvar, geom_hat

    @torch.no_grad()
    def encode(self, rho: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(rho)
        return mu

    @torch.no_grad()
    def decode(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        grid_shape: tuple[int, int] = (128, 128),
    ) -> torch.Tensor:
        return self.decoder(z, t, grid_shape=grid_shape)

    @torch.no_grad()
    def predict_geometry(self, z: torch.Tensor) -> torch.Tensor:
        return self.geom_head.evaluate_at_canonical(z, self.x_canonical)


def vae_loss(
    rho: torch.Tensor,
    rho_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    geom: torch.Tensor,
    geom_hat: torch.Tensor,
    beta: float,
    lambda_geom: float,
    free_bits: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Reconstruction + KL (with free-bits) + geometry MSE."""
    l_rec = F.mse_loss(rho_hat, rho)
    per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_raw = per_dim.mean(0).sum()
    l_kl = per_dim.mean(0).clamp(min=free_bits).sum()
    l_geom = F.mse_loss(geom_hat, geom)
    total = l_rec + beta * l_kl + lambda_geom * l_geom
    return {
        "total": total,
        "rec": l_rec,
        "kl": kl_raw,
        "kl_freebits": l_kl,
        "geom": l_geom,
    }


class BetaScheduler:
    """Linear warmup, then a tiny PID controller targeting a fixed KL budget."""

    def __init__(
        self,
        beta_max: float = 5e-4,
        warmup_epochs: int = 100,
        kl_target: float = 18.0,
        pid_kp: float = 1e-5,
        beta_min: float = 1e-6,
    ) -> None:
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.kl_target = kl_target
        self.pid_kp = pid_kp
        self.beta_min = beta_min
        self._beta = 0.0
        self._epoch = 0

    def step(self, kl_current: float) -> float:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            self._beta = self.beta_max * self._epoch / max(1, self.warmup_epochs)
        else:
            error = kl_current - self.kl_target
            self._beta = float(
                np.clip(self._beta + self.pid_kp * error, self.beta_min, self.beta_max)
            )
        return self._beta

    @property
    def beta(self) -> float:
        return self._beta


def active_units(mu: torch.Tensor, threshold: float = 0.01) -> int:
    """Number of latent dimensions with sample variance above the threshold."""
    return int((mu.var(0) > threshold).sum().item())
