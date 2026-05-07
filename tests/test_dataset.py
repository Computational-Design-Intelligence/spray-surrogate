"""Dataset loader smoke tests using a synthetic HDF5."""
import h5py
import numpy as np
import pytest

from spray_surrogate.dataset import NozzleDataset, points_to_density


@pytest.fixture
def fake_h5(tmp_path):
    path = tmp_path / "fake.h5"
    n_sims = 3
    n_t = 4
    M = 8
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        curves = rng.uniform(-1.0, 1.0, size=(n_sims, M, 2)).astype(np.float32)
        f.create_dataset("curves", data=curves)
        f.create_dataset("x_canonical", data=np.linspace(0.0, 0.7, M).astype(np.float32))
        sims = f.create_group("simulations")
        for s in range(n_sims):
            sg = sims.create_group(f"sim_{s:04d}")
            snap = sg.create_group("snapshots")
            for t_val in np.linspace(0.1, 1.0, n_t):
                tg = snap.create_group(f"t_{t_val:.2f}")
                pts = rng.uniform([0.0, -2.0], [4.0, 2.0], size=(64, 2)).astype(np.float32)
                tg.create_dataset("points", data=pts)
                tg.attrs["time"] = float(t_val)
                tg.attrs["n_cells"] = 64
    return path


def test_points_to_density_shape_and_range():
    pts = np.random.uniform([0.0, -2.0], [4.0, 2.0], size=(100, 2)).astype(np.float32)
    rho = points_to_density(pts, nx=32, ny=32)
    assert rho.shape == (32, 32)
    assert rho.dtype == np.float32
    assert rho.min() >= 0.0
    assert rho.max() <= 1.0 + 1e-6


def test_dataset_basic(fake_h5):
    ds = NozzleDataset(fake_h5, nx=32, ny=32, t_min=0.0, precompute=True)
    assert len(ds) == 3 * 4
    fields, geom, t = ds[0]
    assert fields.shape == (1, 32, 32)
    assert geom.dim() == 1
    assert t.dim() == 0


def test_split_by_simulation_disjoint(fake_h5):
    ds = NozzleDataset(fake_h5, nx=32, ny=32, t_min=0.0, precompute=True)
    sp = ds.split_by_simulation(val_frac=0.34, test_frac=0.34, seed=0)
    train, val, test = set(sp["train_sims"]), set(sp["val_sims"]), set(sp["test_sims"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
