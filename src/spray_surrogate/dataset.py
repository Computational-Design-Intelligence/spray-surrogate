"""HDF5-backed nozzle simulation dataset with cached KDE projections."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Physical domain (must match the data preprocessing pipeline).
X_LO, X_HI = 0.0, 4.0
Y_LO, Y_HI = -2.0, 2.0


def points_to_density(
    pts: np.ndarray,
    nx: int,
    ny: int,
    x_lim: tuple[float, float] = (X_LO, X_HI),
    y_lim: tuple[float, float] = (Y_LO, Y_HI),
    bandwidth: float = 0.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """KDE projection of (N, 2) points to an (nx, ny) grid; log1p-normalised to [0, 1]."""
    x_lo, x_hi = x_lim
    y_lo, y_hi = y_lim
    cell_x = (x_hi - x_lo) / nx
    cell_y = (y_hi - y_lo) / ny
    if bandwidth <= 0:
        bandwidth = 2.0 * max(cell_x, cell_y)

    gx = np.linspace(x_lo + cell_x / 2, x_hi - cell_x / 2, nx)
    gy = np.linspace(y_lo + cell_y / 2, y_hi - cell_y / 2, ny)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    px = pts[:, 0].clip(x_lo, x_hi)
    py = pts[:, 1].clip(y_lo, y_hi)

    rho = np.zeros((nx, ny), dtype=np.float64)
    inv2s2 = 1.0 / (2.0 * bandwidth**2)
    batch = 2048
    for start in range(0, len(px), batch):
        dx = GX[:, :, None] - px[None, None, start : start + batch]
        dy = GY[:, :, None] - py[None, None, start : start + batch]
        rho += np.exp(-(dx**2 + dy**2) * inv2s2).sum(-1)

    rho = np.log1p(rho.astype(np.float32) + eps)
    if rho.max() > 0:
        rho /= rho.max()
    return rho


def points_to_multifield(
    pts: np.ndarray,
    fields: dict[str, np.ndarray],
    nx: int,
    ny: int,
    x_lim: tuple[float, float] = (X_LO, X_HI),
    y_lim: tuple[float, float] = (Y_LO, Y_HI),
    bandwidth: float = 0.0,
) -> np.ndarray:
    """Nadaraya-Watson interpolation of (density + named fields). Returns (1+|fields|, nx, ny)."""
    n_pts = len(pts)
    for k in list(fields):
        if len(fields[k]) != n_pts:
            n_min = min(n_pts, len(fields[k]))
            pts = pts[:n_min]
            fields = {kk: vv[:n_min] for kk, vv in fields.items()}
            n_pts = n_min
            break

    x_lo, x_hi = x_lim
    y_lo, y_hi = y_lim
    cell_x = (x_hi - x_lo) / nx
    cell_y = (y_hi - y_lo) / ny
    if bandwidth <= 0:
        bandwidth = 2.0 * max(cell_x, cell_y)

    gx = np.linspace(x_lo + cell_x / 2, x_hi - cell_x / 2, nx)
    gy = np.linspace(y_lo + cell_y / 2, y_hi - cell_y / 2, ny)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    px = pts[:, 0].clip(x_lo, x_hi)
    py = pts[:, 1].clip(y_lo, y_hi)

    inv2s2 = 1.0 / (2.0 * bandwidth**2)
    weight_sum = np.zeros((nx, ny), dtype=np.float64)
    field_names = sorted(fields)
    field_sums: dict[str, np.ndarray] = {
        k: np.zeros((nx, ny), dtype=np.float64) for k in field_names
    }

    batch = 2048
    for start in range(0, len(px), batch):
        end = min(start + batch, len(px))
        dx = GX[:, :, None] - px[None, None, start:end]
        dy = GY[:, :, None] - py[None, None, start:end]
        w = np.exp(-(dx**2 + dy**2) * inv2s2)
        weight_sum += w.sum(-1)
        for k in field_names:
            vals = fields[k][start:end].astype(np.float64)
            field_sums[k] += (w * vals[None, None, :]).sum(-1)

    density = np.log1p(weight_sum.astype(np.float32) + 1e-3)
    if density.max() > 0:
        density /= density.max()

    weight_safe = np.maximum(weight_sum, 1e-10)
    out = np.zeros((1 + len(field_names), nx, ny), dtype=np.float32)
    out[0] = density
    for ci, k in enumerate(field_names):
        out[ci + 1] = (field_sums[k] / weight_safe).astype(np.float32)
    return out


class _GridCache:
    """On-disk HDF5 cache of pre-computed KDE grids, keyed by a config hash."""

    def __init__(
        self,
        cache_path: Path,
        nx: int,
        ny: int,
        bandwidth: float,
        x_lim: tuple,
        y_lim: tuple,
        t_min: float,
        multifield: bool,
        field_names: Sequence[str],
    ) -> None:
        self.path = cache_path
        key = (
            f"{nx}_{ny}_{bandwidth:.6f}_{x_lim}_{y_lim}_{t_min}_"
            f"{multifield}_{sorted(field_names)}"
        )
        self.tag = hashlib.md5(key.encode()).hexdigest()[:10]

    def is_valid(self, expected_count: int) -> bool:
        if not self.path.exists():
            return False
        try:
            with h5py.File(self.path, "r") as f:
                if self.tag not in f:
                    return False
                count = sum(len(f[self.tag][s]) for s in f[self.tag])
                return count == expected_count
        except Exception:
            return False

    def save(self, grids: dict[str, dict[str, np.ndarray]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "a") as f:
            if self.tag in f:
                del f[self.tag]
            grp = f.create_group(self.tag)
            for sim_key, t_dict in grids.items():
                sg = grp.create_group(sim_key)
                for t_key, arr in t_dict.items():
                    sg.create_dataset(
                        t_key, data=arr, compression="gzip", compression_opts=4
                    )

    def load(self, index: list[tuple[str, str]]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        with h5py.File(self.path, "r") as f:
            grp = f[self.tag]
            for sim_key, t_key in index:
                arr = np.array(grp[sim_key][t_key], dtype=np.float32)
                t = torch.from_numpy(arr)
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                out.append(t)
        return out


class NozzleDataset(Dataset):
    """One sample per (simulation, timestep) snapshot."""

    def __init__(
        self,
        h5_path: str | Path,
        nx: int = 128,
        ny: int = 128,
        t_min: float = 0.1,
        t_max: float = float("inf"),
        bandwidth: float = 0.0,
        x_lim: tuple[float, float] = (X_LO, X_HI),
        y_lim: tuple[float, float] = (Y_LO, Y_HI),
        precompute: bool = True,
        multifield: bool = False,
        field_names: Sequence[str] = ("f", "u_x", "u_y"),
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.nx, self.ny = nx, ny
        self.bandwidth = bandwidth
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.multifield = multifield
        self.field_names = list(field_names)

        with h5py.File(self.h5_path, "r") as f:
            curves_raw = np.array(f["curves"], dtype=np.float32)
            self.x_canonical = torch.from_numpy(np.array(f["x_canonical"], dtype=np.float32))
            sim_keys = sorted(f["simulations"].keys())
            self._sim_keys = sim_keys

            self._index: list[tuple[int, str, str, float]] = []
            for sim_idx, sim_key in enumerate(sim_keys):
                snap_grp = f[f"simulations/{sim_key}/snapshots"]
                for t_key in sorted(snap_grp.keys()):
                    t_val = snap_grp[t_key].attrs.get("time", None)
                    if t_val is None:
                        try:
                            t_val = float(t_key.replace("t_", ""))
                        except ValueError:
                            continue
                    if not (t_min - 1e-9 <= t_val <= t_max + 1e-9):
                        continue
                    if "points" not in snap_grp[t_key]:
                        continue
                    n_cells = snap_grp[t_key].attrs.get(
                        "n_cells", len(snap_grp[t_key]["points"])
                    )
                    if n_cells < 10:
                        continue
                    self._index.append((sim_idx, sim_key, t_key, float(t_val)))

        n_total = len(self._index)
        logger.info(
            "Loaded %d snapshots from %d simulations (h5: %s)",
            n_total,
            len(sim_keys),
            self.h5_path.name,
        )

        # Geometry normalisation, NaN-safe.
        geom_flat = np.concatenate([curves_raw[:, :, 0], curves_raw[:, :, 1]], axis=1)
        self.geom_dim = geom_flat.shape[1]
        self.geom_min = float(np.nanmin(geom_flat))
        self.geom_max = float(np.nanmax(geom_flat))
        geom_norm = (geom_flat - self.geom_min) / (self.geom_max - self.geom_min + 1e-8)
        self.geom_norm = torch.from_numpy(geom_norm.astype(np.float32))

        n_nan = int(np.isnan(geom_flat).any(axis=1).sum())
        if n_nan > 0:
            logger.warning(
                "%d simulation(s) have NaN geometry; filter them at training time.", n_nan
            )

        self.rho_cache: list[torch.Tensor] | None = None
        if precompute:
            self._precompute_grids(t_min)

    def _precompute_grids(self, t_min: float) -> None:
        n_total = len(self._index)
        mf_tag = f"_mf{'_'.join(self.field_names)}" if self.multifield else ""
        cache_path = self.h5_path.parent / f"density_cache_{self.nx}x{self.ny}{mf_tag}.h5"
        cache = _GridCache(
            cache_path,
            self.nx,
            self.ny,
            self.bandwidth,
            self.x_lim,
            self.y_lim,
            t_min,
            self.multifield,
            self.field_names,
        )
        index_keys = [(sk, tk) for (_, sk, tk, _) in self._index]

        if cache.is_valid(n_total):
            self.rho_cache = cache.load(index_keys)
            logger.info("Loaded %d cached grids from %s", n_total, cache_path)
            return

        logger.info("Pre-computing %d grids -> %s", n_total, cache_path)
        flat: list[torch.Tensor] = []
        by_sim: dict[str, dict[str, np.ndarray]] = {}
        with h5py.File(self.h5_path, "r") as f:
            for i, (_, sim_key, t_key, _) in enumerate(self._index):
                snap = f[f"simulations/{sim_key}/snapshots/{t_key}"]
                pts = np.array(snap["points"], dtype=np.float32)
                if self.multifield:
                    fld = {
                        name: (
                            np.array(snap[name], dtype=np.float32)
                            if name in snap
                            else np.zeros(len(pts), dtype=np.float32)
                        )
                        for name in self.field_names
                    }
                    grid = points_to_multifield(
                        pts, fld, self.nx, self.ny, self.x_lim, self.y_lim, self.bandwidth
                    )
                else:
                    rho = points_to_density(
                        pts, self.nx, self.ny, self.x_lim, self.y_lim, self.bandwidth
                    )
                    grid = rho[np.newaxis, ...]
                flat.append(torch.from_numpy(grid))
                by_sim.setdefault(sim_key, {})[t_key] = grid
                if (i + 1) % 500 == 0:
                    logger.info("  %d/%d", i + 1, n_total)
        cache.save(by_sim)
        self.rho_cache = flat

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sim_idx, sim_key, t_key, t_val = self._index[idx]
        if self.rho_cache is not None:
            fields = self.rho_cache[idx]
        else:
            with h5py.File(self.h5_path, "r") as f:
                snap = f[f"simulations/{sim_key}/snapshots/{t_key}"]
                pts = np.array(snap["points"], dtype=np.float32)
                if self.multifield:
                    fld = {
                        name: (
                            np.array(snap[name], dtype=np.float32)
                            if name in snap
                            else np.zeros(len(pts), dtype=np.float32)
                        )
                        for name in self.field_names
                    }
                    grid = points_to_multifield(
                        pts, fld, self.nx, self.ny, self.x_lim, self.y_lim, self.bandwidth
                    )
                    fields = torch.from_numpy(grid)
                else:
                    rho = points_to_density(
                        pts, self.nx, self.ny, self.x_lim, self.y_lim, self.bandwidth
                    )
                    fields = torch.from_numpy(rho).unsqueeze(0)
        geom = self.geom_norm[sim_idx]
        t = torch.tensor(t_val, dtype=torch.float32)
        return fields, geom, t

    def get_sim_info(self, idx: int) -> dict:
        sim_idx, sim_key, t_key, t_val = self._index[idx]
        return {"sim_idx": sim_idx, "sim_key": sim_key, "t_key": t_key, "t_val": t_val}

    def split_by_simulation(
        self,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 0,
    ) -> dict[str, list[int]]:
        """Partition snapshots by simulation id; returns indices for each split."""
        n_sims = len(self._sim_keys)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_sims)
        n_test = int(test_frac * n_sims)
        n_val = int(val_frac * n_sims)
        n_train = n_sims - n_val - n_test
        train = set(perm[:n_train].tolist())
        val = set(perm[n_train : n_train + n_val].tolist())
        test = set(perm[n_train + n_val :].tolist())

        out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for i, (sim_idx, *_) in enumerate(self._index):
            if sim_idx in train:
                out["train"].append(i)
            elif sim_idx in val:
                out["val"].append(i)
            else:
                out["test"].append(i)
        out["train_sims"] = sorted(train)
        out["val_sims"] = sorted(val)
        out["test_sims"] = sorted(test)
        return out
