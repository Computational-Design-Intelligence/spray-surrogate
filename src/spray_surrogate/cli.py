"""Command-line entrypoints exposed via [project.scripts] in pyproject.toml."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from .dataset import NozzleDataset
from .field_predictor import FieldPredictor, field_loss, load_field_predictor
from .inversion import (
    encode_training_sims,
    evaluate_z_on_sim,
    find_topk_in_geometry,
    gather_sim_snapshots,
    invert_geometry,
)
from .vae import FNOVAE, BetaScheduler, active_units, vae_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_run_dir(base_out_dir: str | Path, config_path: str | Path | None) -> Path:
    """Create a timestamped run dir and snapshot the config + git commit."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_out_dir) / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy(config_path, out_dir / "config.yaml")
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        (out_dir / "git_commit.txt").write_text(commit + "\n")
    except Exception:
        pass
    return out_dir


def _parse_common_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--device", default=None, type=str)
    return p.parse_args()


def _load_vae_for_inference(checkpoint_path: str | Path, device: torch.device) -> FNOVAE:
    """Load an FNOVAE for read-only inference. Tolerates old/flat config schemas."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    mcfg = cfg.get("model", cfg)  # support legacy flat configs

    model = FNOVAE(
        latent_dim=mcfg.get("d_z", mcfg.get("latent_dim", 48)),
        d_model=mcfg.get("d_model", 64),
        n_fno_blocks=mcfg.get("n_fno_blocks", mcfg.get("n_fno_layers", 4)),
        n_modes=mcfg.get("n_modes", 16),
        decoder_hidden=mcfg.get("decoder_hidden", mcfg.get("dec_hidden", 128)),
        decoder_layers=mcfg.get("decoder_layers", mcfg.get("dec_layers", 4)),
        decoder_fourier=mcfg.get("decoder_fourier", mcfg.get("dec_fourier", 64)),
        spatial_sigmas=mcfg.get("spatial_sigmas", (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)),
        time_frequencies=mcfg.get("time_frequencies", (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)),
        geom_hidden=mcfg.get("geom_hidden", 64),
        geom_layers=mcfg.get("geom_layers", 3),
        geom_fourier=mcfg.get("geom_fourier", 32),
        x_lo_nozzle=mcfg.get("x_lo_nozzle", 0.0),
        x_hi_nozzle=mcfg.get("x_hi_nozzle", 0.7),
        geom_dim=ck.get("geom_dim", 200),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _save_vae_ckpt(
    path: Path,
    model: FNOVAE,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    loss: float,
    cfg: dict,
    geom_dim: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "cfg": cfg,
            "geom_dim": geom_dim,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        path,
    )


def _filter_valid_geometry(
    geom: torch.Tensor, geom_hat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop rows where the target geometry contains NaNs."""
    valid = ~torch.isnan(geom).any(dim=-1)
    if valid.any():
        return geom[valid], geom_hat[valid]
    # All-NaN batch: emit zero-loss tensors so the gradient is well-defined.
    return torch.zeros_like(geom_hat[:1]), torch.zeros_like(geom_hat[:1])


# ---------------------------------------------------------------------------
# Stage-1 VAE training
# ---------------------------------------------------------------------------


def _vae_step(
    model: FNOVAE,
    fields: torch.Tensor,
    geom: torch.Tensor,
    t: torch.Tensor,
    cfg: dict,
    beta: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rho = fields[:, :1] if fields.dim() == 4 and fields.shape[1] > 1 else fields
    rho_hat, mu, logvar, geom_hat = model(rho, t)
    g_clean, gh_clean = _filter_valid_geometry(geom, geom_hat)
    losses = vae_loss(
        rho,
        rho_hat,
        mu,
        logvar,
        g_clean,
        gh_clean,
        beta=beta,
        lambda_geom=cfg["loss"]["lambda_geom"],
        free_bits=cfg["loss"]["free_bits"],
    )
    return losses, mu


def _run_vae_training(
    model: FNOVAE,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    geom_dim: int,
) -> None:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["optim"]["epochs"], eta_min=cfg["optim"]["lr"] * 0.01
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    beta_s = BetaScheduler(
        beta_max=cfg["loss"]["beta_max"],
        warmup_epochs=cfg["loss"]["beta_warmup"],
        kl_target=cfg["loss"]["kl_target"],
        pid_kp=cfg["loss"]["pid_kp"],
    )
    writer = SummaryWriter(out_dir / "tb")
    best_score = float("inf")
    tr_stats: dict[str, float] = {"kl": 0.0}

    for epoch in range(cfg["optim"]["epochs"]):
        t0 = time.perf_counter()
        beta = beta_s.step(0.0 if epoch == 0 else tr_stats["kl"])

        model.train()
        totals: dict[str, float] = defaultdict(float)
        all_mu: list[torch.Tensor] = []
        for fields, geom, t in train_dl:
            fields = fields.to(device)
            geom = geom.to(device)
            t = t.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                losses, mu = _vae_step(model, fields, geom, t, cfg, beta)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            scaler.step(opt)
            scaler.update()

            for k, v in losses.items():
                totals[k] += v.item()
            all_mu.append(mu.detach().cpu())

        n_tr = len(train_dl)
        tr_stats = {k: v / n_tr for k, v in totals.items()}
        au = active_units(torch.cat(all_mu))
        sched.step()

        # Validation
        model.eval()
        v_totals: dict[str, float] = defaultdict(float)
        with torch.no_grad():
            for fields, geom, t in val_dl:
                fields = fields.to(device)
                geom = geom.to(device)
                t = t.to(device)
                losses, _ = _vae_step(model, fields, geom, t, cfg, beta)
                for k, v in losses.items():
                    v_totals[k] += v.item()
        n_va = max(len(val_dl), 1)
        va_stats = {k: v / n_va for k, v in v_totals.items()}
        score = va_stats["rec"] + va_stats["geom"]

        if epoch % cfg["output"]["log_every"] == 0 or epoch == cfg["optim"]["epochs"] - 1:
            logger.info(
                "[%05d/%d] beta=%.2e rec=%.5f kl=%.2f geom=%.5f "
                "val_rec=%.5f val_geom=%.5f AU=%d/%d %.1fs",
                epoch,
                cfg["optim"]["epochs"],
                beta,
                tr_stats["rec"],
                tr_stats["kl"],
                tr_stats["geom"],
                va_stats["rec"],
                va_stats["geom"],
                au,
                cfg["model"]["d_z"],
                time.perf_counter() - t0,
            )

        for k, v in tr_stats.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in va_stats.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("beta", beta, epoch)
        writer.add_scalar("active_units", au, epoch)

        if score < best_score:
            best_score = score
            _save_vae_ckpt(
                out_dir / "best.pt", model, opt, scaler, epoch, va_stats["rec"], cfg, geom_dim
            )
        if (
            cfg["output"].get("ckpt_every", 0) > 0
            and epoch > 0
            and epoch % cfg["output"]["ckpt_every"] == 0
        ):
            _save_vae_ckpt(
                out_dir / f"ckpt_{epoch:05d}.pt",
                model,
                opt,
                scaler,
                epoch,
                va_stats["rec"],
                cfg,
                geom_dim,
            )

    _save_vae_ckpt(
        out_dir / "final.pt",
        model,
        opt,
        scaler,
        cfg["optim"]["epochs"] - 1,
        best_score,
        cfg,
        geom_dim,
    )
    writer.close()
    logger.info("Training complete. Best score: %.6f", best_score)


def train_vae() -> None:
    configure_logging()
    args = _parse_common_cli()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg["device"])
    logger.info("Device: %s", device)

    out_dir = setup_run_dir(cfg["output"]["out_dir"], args.config)

    ds = NozzleDataset(
        h5_path=cfg["data"]["path"],
        nx=cfg["data"]["nx"],
        ny=cfg["data"]["ny"],
        t_min=cfg["data"]["t_min"],
        precompute=True,
        multifield=cfg["data"].get("multifield", False),
        field_names=cfg["data"].get("field_names", ("f", "u_x", "u_y")),
    )
    splits = ds.split_by_simulation(
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        seed=cfg.get("seed", 0),
    )
    torch.save(splits, out_dir / "split_info.pt")
    logger.info(
        "Split: %d train / %d val / %d test snapshots",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    train_dl = DataLoader(
        Subset(ds, splits["train"]),
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_dl = DataLoader(
        Subset(ds, splits["val"]),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = FNOVAE(
        latent_dim=cfg["model"]["d_z"],
        d_model=cfg["model"]["d_model"],
        n_fno_blocks=cfg["model"]["n_fno_blocks"],
        n_modes=cfg["model"]["n_modes"],
        decoder_hidden=cfg["model"]["decoder_hidden"],
        decoder_layers=cfg["model"]["decoder_layers"],
        decoder_fourier=cfg["model"]["decoder_fourier"],
        spatial_sigmas=cfg["model"]["spatial_sigmas"],
        time_frequencies=cfg["model"]["time_frequencies"],
        geom_hidden=cfg["model"]["geom_hidden"],
        geom_layers=cfg["model"]["geom_layers"],
        geom_fourier=cfg["model"]["geom_fourier"],
        x_lo_nozzle=cfg["model"]["x_lo_nozzle"],
        x_hi_nozzle=cfg["model"]["x_hi_nozzle"],
        geom_dim=ds.geom_dim,
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("FNOVAE: %d parameters", n_params)

    _run_vae_training(model, train_dl, val_dl, cfg, device, out_dir, ds.geom_dim)


# ---------------------------------------------------------------------------
# Stage-2 field predictor training
# ---------------------------------------------------------------------------


@torch.no_grad()
def _precompute_rho_hat(
    ds: NozzleDataset, vae: FNOVAE, cache_path: Path, device: torch.device
) -> torch.Tensor:
    if cache_path.exists():
        logger.info("Loading rho_hat cache from %s", cache_path)
        return torch.load(cache_path, weights_only=True)
    logger.info("Pre-computing rho_hat for %d snapshots", len(ds))
    Nx, Ny = ds.nx, ds.ny
    out = torch.zeros(len(ds), 1, Nx, Ny, dtype=torch.float32)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    idx = 0
    t0 = time.perf_counter()
    for fields, _, t in loader:
        if fields.dim() == 4:
            rho = fields[:, :1].to(device)
        else:
            rho = fields.unsqueeze(1).to(device)
        mu, _ = vae.encoder(rho)
        rh = vae.decoder(mu, t.to(device), grid_shape=(Nx, Ny))
        B = rh.shape[0]
        out[idx : idx + B] = rh.cpu()
        idx += B
        if idx % 1000 < B:
            rate = idx / max(time.perf_counter() - t0, 1e-6)
            logger.info("  %d/%d  (%.0f samples/s)", idx, len(ds), rate)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    return out


class _FieldDataset(Dataset):
    """(density_input, [f, u, v]) pairs for Stage-2 training."""

    def __init__(
        self,
        base: NozzleDataset,
        indices: list[int],
        rho_hat: torch.Tensor | None,
    ) -> None:
        self.base = base
        self.indices = indices
        self.rho_hat = rho_hat

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        gi = self.indices[idx]
        fields, _, _ = self.base[gi]
        targets = fields[1:]
        density = self.rho_hat[gi] if self.rho_hat is not None else fields[:1]
        return density, targets


def train_field_predictor() -> None:
    configure_logging()
    args = _parse_common_cli()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg["device"])
    out_dir = setup_run_dir(cfg["output"]["out_dir"], args.config)

    ds = NozzleDataset(
        h5_path=cfg["data"]["path"],
        nx=cfg["data"]["nx"],
        ny=cfg["data"]["ny"],
        t_min=cfg["data"]["t_min"],
        precompute=True,
        multifield=True,
        field_names=("f", "u_x", "u_y"),
    )

    split_path = cfg["input"].get("split_info", "")
    if split_path and Path(split_path).exists():
        splits = torch.load(split_path, weights_only=False)
        train_idx = splits["train"]
        val_idx = splits["val"]
        logger.info(
            "Using existing VAE split (%d train, %d val)", len(train_idx), len(val_idx)
        )
    else:
        splits = ds.split_by_simulation(seed=cfg.get("seed", 0))
        train_idx = splits["train"]
        val_idx = splits["val"]
        logger.warning(
            "No split_info supplied; created a fresh split. This will NOT match VAE training."
        )

    rho_hat: torch.Tensor | None = None
    vae_ckpt = cfg["input"].get("vae_checkpoint", "")
    if vae_ckpt and Path(vae_ckpt).exists():
        vae = _load_vae_for_inference(vae_ckpt, device)
        rho_hat = _precompute_rho_hat(ds, vae, out_dir / "rho_hat_cache.pt", device)
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Training on rho_hat (VAE-reconstructed density)")
    else:
        logger.info("Training on ground-truth density rho")

    train_dl = DataLoader(
        _FieldDataset(ds, train_idx, rho_hat),
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_dl = DataLoader(
        _FieldDataset(ds, val_idx, rho_hat),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = FieldPredictor(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
    ).to(device)
    logger.info("FieldPredictor: %d parameters", sum(p.numel() for p in model.parameters()))

    cw = cfg["loss"]["channel_weights"]
    weights = {0: cw["f"], 1: cw["u"], 2: cw["v"]}

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["optim"]["epochs"], eta_min=cfg["optim"]["lr"] * 0.01
    )

    best_val = float("inf")
    for epoch in range(cfg["optim"]["epochs"]):
        t0 = time.perf_counter()
        model.train()
        tr_totals: dict[str, float] = defaultdict(float)
        for density, targets in train_dl:
            density = density.to(device)
            targets = targets.to(device)
            opt.zero_grad()
            pred = model(density)
            losses = field_loss(pred, targets, weights)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            opt.step()
            for k, v in losses.items():
                tr_totals[k] += v.item()

        n_tr = len(train_dl)
        tr = {k: v / n_tr for k, v in tr_totals.items()}

        model.eval()
        va_totals: dict[str, float] = defaultdict(float)
        with torch.no_grad():
            for density, targets in val_dl:
                density = density.to(device)
                targets = targets.to(device)
                pred = model(density)
                losses = field_loss(pred, targets, weights)
                for k, v in losses.items():
                    va_totals[k] += v.item()
        n_va = max(len(val_dl), 1)
        va = {k: v / n_va for k, v in va_totals.items()}
        sched.step()

        if epoch % cfg["output"]["log_every"] == 0 or epoch == cfg["optim"]["epochs"] - 1:
            logger.info(
                "[%04d/%d] tr f=%.5f u=%.5f v=%.5f | va f=%.5f u=%.5f v=%.5f | %.1fs",
                epoch,
                cfg["optim"]["epochs"],
                tr["f"],
                tr["u"],
                tr["v"],
                va["f"],
                va["u"],
                va["v"],
                time.perf_counter() - t0,
            )

        if va["total"] < best_val:
            best_val = va["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": va,
                    "cfg": {
                        "in_channels": cfg["model"]["in_channels"],
                        "out_channels": cfg["model"]["out_channels"],
                        "base_channels": cfg["model"]["base_channels"],
                        "depth": cfg["model"]["depth"],
                        "trained_on_vae_recon": rho_hat is not None,
                    },
                },
                out_dir / "best.pt",
            )
    logger.info("Done. Best val loss: %.6f", best_val)


# ---------------------------------------------------------------------------
# Geometry-targeted inversion
# ---------------------------------------------------------------------------


def _aggregate_inversion_rows(rows: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    summary: dict = {}
    for method, lst in by_method.items():
        keys = [
            k
            for k in lst[0]
            if k not in ("method", "sim_idx") and isinstance(lst[0][k], int | float)
        ]
        agg: dict = {}
        for key in keys:
            vals = np.array([r[key] for r in lst], dtype=float)
            agg[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(np.median(vals)),
            }
        summary[method] = agg
    return summary


def run_inversion() -> None:
    configure_logging()
    args = _parse_common_cli()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg["device"])
    out_dir = setup_run_dir(cfg["output"]["out_dir"], args.config)

    model = _load_vae_for_inference(cfg["checkpoints"]["vae"], device)
    splits = torch.load(cfg["checkpoints"]["split_info"], weights_only=False)
    train_sim_set = {int(s) for s in splits["train_sims"]}
    test_sims = [int(s) for s in splits["test_sims"]]

    multifield = cfg["data"].get("multifield", False)
    ds = NozzleDataset(
        h5_path=cfg["data"]["path"],
        nx=cfg["data"]["nx"],
        ny=cfg["data"]["ny"],
        t_min=cfg["data"]["t_min"],
        precompute=True,
        multifield=multifield,
        field_names=("f", "u_x", "u_y") if multifield else (),
    )

    sim_mean_z, train_geoms, _ = encode_training_sims(model, ds, train_sim_set, device)
    suggested = float(sim_mean_z.norm(dim=1).max().item()) * 1.1
    if cfg["inversion"]["max_z_norm"] < suggested:
        logger.info(
            "Bumping max_z_norm: %.2f -> %.2f",
            cfg["inversion"]["max_z_norm"],
            suggested,
        )
        cfg["inversion"]["max_z_norm"] = suggested

    stage2 = None
    fp_ckpt = cfg["checkpoints"].get("field_predictor")
    if fp_ckpt:
        try:
            stage2 = load_field_predictor(fp_ckpt, device=str(device))
            logger.info("Stage-2 loaded; f/u/v metrics will be reported")
        except Exception as e:
            logger.warning("Stage-2 load failed (%s); continuing without it", e)

    n_max = cfg["output"].get("max_test_sims", -1)
    test_iter = test_sims if n_max < 0 else test_sims[:n_max]
    rows: list[dict] = []

    for k, sim_idx in enumerate(test_iter):
        snaps = gather_sim_snapshots(ds, sim_idx, multifield=multifield)
        if not snaps:
            continue

        target_g_phys = snaps[0]["geom_phys"]
        target_g_norm = torch.tensor(
            (target_g_phys - ds.geom_min) / (ds.geom_max - ds.geom_min + 1e-8),
            dtype=torch.float32,
        )

        # 1) argmin: geometry-targeted inversion anchored to K nearest training sims.
        anchor_pos, _ = find_topk_in_geometry(
            target_g_phys, train_geoms, k=cfg["inversion"]["n_starts"]
        )
        best = invert_geometry(
            model,
            target_g_norm,
            sim_mean_z,
            anchor_pos,
            n_steps=cfg["inversion"]["n_steps"],
            lr=cfg["inversion"]["lr"],
            max_z_norm=cfg["inversion"]["max_z_norm"],
            lambda_prior=cfg["inversion"]["lambda_prior"],
            device=device,
        )
        m_arg = evaluate_z_on_sim(
            model, best["z"], snaps, ds.geom_min, ds.geom_max, ds.nx, ds.ny, stage2, device
        )
        m_arg.update(method="argmin", sim_idx=int(sim_idx))

        # 2) encoder oracle: uses ground-truth density at t_max.
        rho_late = snaps[-1]["rho_true"].to(device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mu_oracle, _ = model.encoder(rho_late)
        m_orc = evaluate_z_on_sim(
            model, mu_oracle, snaps, ds.geom_min, ds.geom_max, ds.nx, ds.ny, stage2, device
        )
        m_orc.update(method="oracle", sim_idx=int(sim_idx))

        # 3) NN-in-geometry baseline.
        d = np.linalg.norm(train_geoms - target_g_phys[None, :], axis=1)
        nn_pos = int(np.argmin(d))
        z_nn = sim_mean_z[nn_pos].to(device).unsqueeze(0)
        m_nn = evaluate_z_on_sim(
            model, z_nn, snaps, ds.geom_min, ds.geom_max, ds.nx, ds.ny, stage2, device
        )
        m_nn.update(method="nn_geom", sim_idx=int(sim_idx))

        rows.extend([m_arg, m_orc, m_nn])

        if k % 5 == 0 or k == len(test_iter) - 1:
            logger.info(
                "[%d/%d] sim %d  argmin geom_mae=%.4f  rho MSE arg=%.5f orc=%.5f nn=%.5f",
                k + 1,
                len(test_iter),
                sim_idx,
                m_arg["geom_mae"],
                m_arg["rho_mse"],
                m_orc["rho_mse"],
                m_nn["rho_mse"],
            )

    summary = _aggregate_inversion_rows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "rows.npy", np.array(rows, dtype=object))
    logger.info("Saved summary -> %s", out_dir / "summary.json")

    for method in ("oracle", "argmin", "nn_geom"):
        if method not in summary:
            continue
        g = summary[method]["geom_mae"]
        r = summary[method]["rho_mse"]
        psnr = summary[method]["psnr"]
        logger.info(
            "%-10s geom_mae=%.4f+/-%.4f  rho_mse=%.5f+/-%.5f  PSNR=%.2f+/-%.2f",
            method,
            g["mean"],
            g["std"],
            r["mean"],
            r["std"],
            psnr["mean"],
            psnr["std"],
        )
