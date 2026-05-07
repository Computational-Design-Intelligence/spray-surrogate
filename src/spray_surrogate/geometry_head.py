"""1D INR mapping z to nozzle wall functions (y_upper(x), y_lower(x))."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FiLM, SirenLayer


class GeometryHead(nn.Module):
    """z -> (y_upper(x), y_lower(x)) for any x in [x_lo, x_hi]."""

    def __init__(
        self,
        latent_dim: int = 48,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_fourier: int = 32,
        sigma_fourier: float = 5.0,
        x_lo: float = 0.0,
        x_hi: float = 0.7,
    ) -> None:
        super().__init__()
        self.x_lo = x_lo
        self.x_hi = x_hi
        ff_dim = 2 * n_fourier

        # Fixed (non-learned) random Fourier features for the 1D coordinate.
        B_mat = torch.randn(1, n_fourier) * sigma_fourier
        self.register_buffer("B_mat", B_mat)

        self.input_layer = SirenLayer(ff_dim, hidden_dim, omega=30.0, is_first=True)
        self.siren_layers = nn.ModuleList(
            [SirenLayer(hidden_dim, hidden_dim, omega=1.0) for _ in range(n_layers - 1)]
        )
        self.film_layers = nn.ModuleList(
            [FiLM(latent_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.out = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B_mat
        return torch.cat([proj.sin(), proj.cos()], dim=-1)

    def _normalize_x(self, x_phys: torch.Tensor) -> torch.Tensor:
        return (x_phys - self.x_lo) / (self.x_hi - self.x_lo + 1e-8)

    def forward_x(self, z: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        """(B, dz), (N,) or (B, N) -> (B, N, 2) [upper, lower] (normalised)."""
        if x_coords.dim() == 1:
            x_coords = x_coords.unsqueeze(0).expand(z.size(0), -1)
        _, N = x_coords.shape
        x_norm = self._normalize_x(x_coords).unsqueeze(-1)
        ff = self._fourier_features(x_norm)

        h = self.input_layer(ff)
        h = self.film_layers[0](h, z.unsqueeze(1).expand(-1, N, -1))
        for siren, film in zip(self.siren_layers, self.film_layers[1:], strict=False):
            h = siren(h)
            h = film(h, z.unsqueeze(1).expand(-1, N, -1))
        return self.out(h)

    def evaluate_at_canonical(
        self, z: torch.Tensor, x_canonical: torch.Tensor
    ) -> torch.Tensor:
        """Returns (B, 2M) flat [upper..., lower...] for downstream loss compatibility."""
        out = self.forward_x(z, x_canonical.to(z.device))
        return torch.cat([out[..., 0], out[..., 1]], dim=-1)
