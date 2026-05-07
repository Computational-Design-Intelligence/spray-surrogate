"""Geometry-targeted latent inversion.

For each test simulation, find z* such that the geometry head predicts the
target nozzle walls — minimising
    L(z) = || G(z) - g* ||^2 + lambda * || z - z_anchor ||^2
under ||z|| <= z_max. Multi-start with anchors at the K geometry-nearest
training simulations.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def invert_geometry(
    model,
    target_geom_norm: torch.Tensor,
    sim_mean_z: torch.Tensor,
    anchor_indices: list[int],
    n_steps: int = 300,
    lr: float = 5e-2,
    max_z_norm: float = 8.0,
    lambda_prior: float = 1e-3,
    device: torch.device | None = None,
) -> dict:
    """K-restart Adam descent on geom MSE + lambda * ||z - z_anchor||^2.

    `model` must expose `.geom_head` and `.x_canonical` (FNOVAE does).
    Picks the restart with the smallest *combined* loss to avoid selecting
    a starting point that simply over-fit the geometry head.
    """
    target = target_geom_norm.to(device).reshape(1, -1)
    x_can = model.x_canonical
    best: dict = {
        "z": None,
        "total_loss": float("inf"),
        "geom_mse": float("inf"),
        "prior": float("inf"),
        "start_idx": -1,
    }

    for sim_i in anchor_indices:
        z_anchor = sim_mean_z[int(sim_i) : int(sim_i) + 1].clone().to(device)
        z = z_anchor.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr * 0.01
        )

        for _ in range(n_steps):
            opt.zero_grad()
            g_pred = model.geom_head.evaluate_at_canonical(z, x_can)
            geom_mse = F.mse_loss(g_pred, target)
            prior = (z - z_anchor).pow(2).mean()
            loss = geom_mse + lambda_prior * prior
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                zn = z.norm()
                if zn > max_z_norm:
                    z.mul_(max_z_norm / zn)

        with torch.no_grad():
            g_pred = model.geom_head.evaluate_at_canonical(z, x_can)
            geom_mse_v = F.mse_loss(g_pred, target).item()
            prior_v = (z - z_anchor).pow(2).mean().item()
            total_v = geom_mse_v + lambda_prior * prior_v

        if total_v < best["total_loss"]:
            best.update(
                z=z.detach().clone(),
                total_loss=total_v,
                geom_mse=geom_mse_v,
                prior=prior_v,
                start_idx=int(sim_i),
            )
    return best


@torch.no_grad()
def evaluate_z_on_sim(
    model,
    z: torch.Tensor,
    sim_snapshots: list[dict],
    geom_min: float,
    geom_max: float,
    nx: int,
    ny: int,
    stage2: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> dict:
    """Decode z at every snapshot and report density (and optional Stage-2) errors."""
    g_pred_norm = model.geom_head.evaluate_at_canonical(z, model.x_canonical)
    g_pred_phys = g_pred_norm.cpu().numpy()[0] * (geom_max - geom_min) + geom_min
    g_true_phys = sim_snapshots[0]["geom_phys"]
    geom_mae = float(np.abs(g_pred_phys - g_true_phys).mean())

    rho_mses: list[float] = []
    psnrs: list[float] = []
    f_mses: list[float] = []
    u_mses: list[float] = []
    v_mses: list[float] = []

    for snap in sim_snapshots:
        t = torch.tensor([snap["t_val"]], device=device)
        rho_pred = model.decode(z, t, grid_shape=(nx, ny))[0, 0]
        rho_true = snap["rho_true"].to(device)
        mse = F.mse_loss(rho_pred, rho_true).item()
        rho_mses.append(mse)
        psnrs.append(-10.0 * np.log10(max(mse, 1e-10)))

        if stage2 is not None and snap.get("fields_true") is not None:
            fp = stage2(rho_pred.unsqueeze(0).unsqueeze(0))[0]
            ft = snap["fields_true"].to(device)
            f_mses.append(F.mse_loss(fp[0], ft[0]).item())
            u_mses.append(F.mse_loss(fp[1], ft[1]).item())
            v_mses.append(F.mse_loss(fp[2], ft[2]).item())

    out: dict = {
        "geom_mae": geom_mae,
        "rho_mse": float(np.mean(rho_mses)),
        "psnr": float(np.mean(psnrs)),
        "n_snaps": len(sim_snapshots),
    }
    if f_mses:
        out["f_mse"] = float(np.mean(f_mses))
        out["u_mse"] = float(np.mean(u_mses))
        out["v_mse"] = float(np.mean(v_mses))
    return out


def gather_sim_snapshots(ds, sim_idx: int, multifield: bool = False) -> list[dict]:
    """Collect every snapshot of a simulation, sorted by time. Skips NaN-geom sims."""
    g_norm = ds.geom_norm[sim_idx].numpy()
    if np.isnan(g_norm).any():
        return []
    geom_phys = g_norm * (ds.geom_max - ds.geom_min) + ds.geom_min

    snaps: list[dict] = []
    for i in range(len(ds)):
        info = ds.get_sim_info(i)
        if info["sim_idx"] != sim_idx:
            continue
        item = ds[i]
        x = item[0]
        if multifield and x.dim() == 3 and x.shape[0] > 1:
            rho = x[:1]
            fields_true = x[1:]
        else:
            rho = x if x.dim() == 2 else x[:1]
            fields_true = None
        snaps.append(
            {
                "rho_true": rho.squeeze(0) if rho.dim() == 3 else rho,
                "t_val": float(info["t_val"]),
                "geom_phys": geom_phys,
                "fields_true": fields_true,
            }
        )
    snaps.sort(key=lambda d: d["t_val"])
    return snaps


@torch.no_grad()
def encode_training_sims(
    model,
    ds,
    train_sim_set: set[int],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, np.ndarray, list[int]]:
    """Mean encoder mu per training sim; physical-coordinate target geometries."""
    by_sim: dict[int, list[torch.Tensor]] = defaultdict(list)
    geom_by_sim: dict[int, np.ndarray] = {}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    pos = 0
    for batch in loader:
        rho = batch[0]
        if rho.dim() == 4 and rho.shape[1] > 1:
            rho = rho[:, :1]
        mu, _ = model.encoder(rho.to(device))
        for k in range(mu.shape[0]):
            sidx = ds.get_sim_info(pos + k)["sim_idx"]
            if sidx not in train_sim_set:
                continue
            if sidx not in geom_by_sim:
                g_norm = ds.geom_norm[sidx].numpy()
                if np.isnan(g_norm).any():
                    continue
                geom_by_sim[sidx] = g_norm * (ds.geom_max - ds.geom_min) + ds.geom_min
            if sidx in geom_by_sim:
                by_sim[sidx].append(mu[k].cpu())
        pos += rho.shape[0]

    keys = sorted(by_sim.keys())
    sim_mean_z = torch.stack([torch.stack(by_sim[k]).mean(0) for k in keys])
    train_geoms = np.stack([geom_by_sim[k] for k in keys])
    return sim_mean_z, train_geoms, keys


def find_topk_in_geometry(
    target_geom_phys: np.ndarray, train_geoms: np.ndarray, k: int = 5
) -> tuple[list[int], list[float]]:
    """Top-k positions in `train_geoms` by L2 curve distance, ascending."""
    d = np.linalg.norm(train_geoms - target_geom_phys[None, :], axis=1)
    pos = np.argsort(d)[:k]
    return pos.tolist(), d[pos].tolist()
