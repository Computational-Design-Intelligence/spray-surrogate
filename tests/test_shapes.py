"""Forward-pass shape and invariant checks."""
import torch

from spray_surrogate.decoder import TimeConditionedDecoder
from spray_surrogate.encoder import FNOEncoder
from spray_surrogate.field_predictor import FieldPredictor
from spray_surrogate.geometry_head import GeometryHead
from spray_surrogate.vae import FNOVAE, vae_loss


def test_encoder_shapes(density_batch):
    enc = FNOEncoder(latent_dim=48, d_model=64, n_layers=4, n_modes_x=16, n_modes_y=16)
    mu, logvar = enc(density_batch)
    assert mu.shape == (2, 48)
    assert logvar.shape == (2, 48)


def test_decoder_shapes(latent_batch, device):
    dec = TimeConditionedDecoder(latent_dim=48, hidden_dim=128, n_layers=4)
    t = torch.tensor([0.5, 2.0], device=device)
    out = dec(latent_batch, t, grid_shape=(64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert (out >= 0).all(), "Softplus output must be non-negative."


def test_geometry_head_shapes(latent_batch, device):
    head = GeometryHead(latent_dim=48, hidden_dim=64)
    x_q = torch.linspace(0.0, 0.7, 32, device=device)
    walls = head.forward_x(latent_batch, x_q)
    assert walls.shape == (2, 32, 2)
    flat = head.evaluate_at_canonical(latent_batch, x_q)
    assert flat.shape == (2, 64)


def test_field_predictor_shapes(device):
    model = FieldPredictor(in_channels=1, out_channels=3, base_channels=16, depth=3)
    x = torch.randn(2, 1, 128, 128, device=device)
    out = model(x)
    assert out.shape == (2, 3, 128, 128)
    # Channel 0 (volume fraction f) must be sigmoid-bounded.
    assert ((out[:, 0] >= 0.0) & (out[:, 0] <= 1.0)).all()


def test_vae_forward(density_batch, device):
    model = FNOVAE(latent_dim=48, geom_dim=200)
    t = torch.tensor([1.0, 2.0], device=device)
    rho_hat, mu, logvar, geom_hat = model(density_batch, t)
    assert rho_hat.shape == density_batch.shape
    assert mu.shape == (2, 48)
    assert logvar.shape == (2, 48)
    assert geom_hat.shape == (2, 200)


def test_vae_loss_components(density_batch, device):
    model = FNOVAE(latent_dim=48, geom_dim=200)
    t = torch.tensor([1.0, 2.0], device=device)
    rho_hat, mu, logvar, geom_hat = model(density_batch, t)
    geom = torch.randn(2, 200, device=device)
    losses = vae_loss(
        density_batch, rho_hat, mu, logvar, geom, geom_hat,
        beta=1e-4, lambda_geom=5.0, free_bits=0.5,
    )
    for k in ("total", "rec", "kl", "kl_freebits", "geom"):
        assert k in losses
    assert losses["total"].dim() == 0


def test_decoder_mesh_invariance(latent_batch, device):
    """Decoder should accept different grid resolutions without error."""
    dec = TimeConditionedDecoder(latent_dim=48, hidden_dim=128, n_layers=4)
    t = torch.tensor([0.5, 0.5], device=device)
    for shape in [(32, 32), (64, 64), (128, 128)]:
        out = dec(latent_batch, t, grid_shape=shape)
        assert out.shape == (2, 1, *shape)
