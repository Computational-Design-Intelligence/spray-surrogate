"""FNO encoder: density field rho(x, y) -> latent (mu, log var)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """2D Fourier integral operator: truncate, multiply, inverse-FFT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_x: int = 16,
        n_modes_y: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        scale = 1.0 / (in_channels * out_channels)
        # Stored as float pairs (real, imag) so an old DDP/checkpoint that
        # reads them as float32 still loads cleanly; we view as complex on use.
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, 2)
        )

    @staticmethod
    def _complex_mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, Nx, Ny = x.shape
        in_dtype = x.dtype
        # FFT requires float32 — cast up if AMP downcast to float16, restore at end.
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")

        out_ft = torch.zeros(
            B, self.out_channels, Nx, Ny // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        mx = min(self.n_modes_x, Nx)
        my = min(self.n_modes_y, Ny // 2 + 1)
        w_c = torch.view_as_complex(self.weights.float().contiguous())

        out_ft[:, :, :mx, :my] = self._complex_mul(
            x_ft[:, :, :mx, :my], w_c[:, :, :mx, :my]
        )
        out_ft[:, :, -mx:, :my] = self._complex_mul(
            x_ft[:, :, -mx:, :my], w_c[:, :, :mx, :my]
        )
        out = torch.fft.irfft2(out_ft, s=(Nx, Ny), norm="ortho")
        return out.to(in_dtype)


class FNOBlock(nn.Module):
    """One FNO block: spectral conv + 1x1 bypass + GroupNorm + GELU."""

    def __init__(self, d_model: int, n_modes_x: int = 16, n_modes_y: int = 16) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(d_model, d_model, n_modes_x, n_modes_y)
        self.bypass = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.GroupNorm(min(8, d_model), d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class FNOEncoder(nn.Module):
    """rho(x, y) -> (mu, log var). Adds two normalised coordinate channels."""

    def __init__(
        self,
        latent_dim: int = 48,
        d_model: int = 64,
        n_layers: int = 4,
        n_modes_x: int = 16,
        n_modes_y: int = 16,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(3, d_model, 1)
        self.fno_blocks = nn.Sequential(
            *[FNOBlock(d_model, n_modes_x, n_modes_y) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 1),
            nn.GELU(),
        )
        self.to_mu = nn.Linear(d_model // 2, latent_dim)
        self.to_logvar = nn.Linear(d_model // 2, latent_dim)
        nn.init.zeros_(self.to_mu.weight)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -2.0)

    @staticmethod
    def _coord_grid(B: int, Nx: int, Ny: int, device: torch.device) -> torch.Tensor:
        xs = torch.linspace(0.0, 1.0, Nx, device=device)
        ys = torch.linspace(0.0, 1.0, Ny, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([gx, gy], dim=0)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, Nx, Ny = rho.shape
        coords = self._coord_grid(B, Nx, Ny, rho.device)
        x = torch.cat([rho, coords], dim=1)
        x = self.lift(x)
        x = self.fno_blocks(x)
        x = self.proj(x)
        x = x.mean(dim=(-1, -2))
        mu = self.to_mu(x)
        logvar = self.to_logvar(x).clamp(-6.0, 2.0)
        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        if training:
            return mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return mu
