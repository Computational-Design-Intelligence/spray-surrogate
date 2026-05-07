"""Geometry-targeted inversion sanity check."""
import torch
import torch.nn.functional as F

from spray_surrogate.geometry_head import GeometryHead
from spray_surrogate.inversion import find_topk_in_geometry, invert_geometry


class _StubVAE:
    """Minimal stand-in for an FNOVAE — only the attributes inversion uses."""

    def __init__(self, head: GeometryHead, x_canonical: torch.Tensor):
        self.geom_head = head
        self.x_canonical = x_canonical


def _make_nontrivial_head(latent_dim: int = 48, hidden_dim: int = 64) -> GeometryHead:
    """Replace the zero-initialised output layer with small random weights so
    that the head implements a non-degenerate function we can invert against."""
    head = GeometryHead(latent_dim=latent_dim, hidden_dim=hidden_dim).eval()
    torch.nn.init.uniform_(head.out.weight, -0.1, 0.1)
    torch.nn.init.uniform_(head.out.bias, -0.1, 0.1)
    return head


def test_invert_reduces_objective(device):
    torch.manual_seed(0)
    head = _make_nontrivial_head()
    x_can = torch.linspace(0.0, 0.7, 100, device=device)

    z_true = torch.randn(48, device=device)
    with torch.no_grad():
        target = head.evaluate_at_canonical(z_true.unsqueeze(0), x_can).squeeze(0)

    sim_mean_z = torch.randn(20, 48, device=device)
    stub = _StubVAE(head, x_can)

    best = invert_geometry(
        stub, target, sim_mean_z, anchor_indices=[0, 5, 10],
        n_steps=100, lr=5e-2, max_z_norm=10.0, lambda_prior=1e-3, device=device,
    )
    assert best["z"] is not None

    # Initial residual taken from the best anchor so this can't trivially pass
    # by descending from a random one.
    with torch.no_grad():
        anchor_mses = [
            F.mse_loss(
                head.evaluate_at_canonical(sim_mean_z[i : i + 1], x_can),
                target.unsqueeze(0),
            ).item()
            for i in (0, 5, 10)
        ]
    assert best["geom_mse"] < min(anchor_mses)





def test_topk_geometry_returns_sorted():
    import numpy as np
    train_geoms = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 3.0]])
    target = np.array([0.1, 0.0])
    pos, dists = find_topk_in_geometry(target, train_geoms, k=3)
    assert pos == [0, 1, 2]
    assert dists == sorted(dists)
