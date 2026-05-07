import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def density_batch(device):
    return torch.randn(2, 1, 128, 128, device=device)


@pytest.fixture
def latent_batch(device):
    return torch.randn(2, 48, device=device)
