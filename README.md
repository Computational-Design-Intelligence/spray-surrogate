# Spray Surrogate

Geometry-conditioned latent surrogate for two-phase nozzle flow.
Code accompanying the AI4Physics @ ICML 2026 submission.

## Setup

```bash
conda env create -f environment.yml
conda activate spray-surrogate
pre-commit install   # optional but recommended
```

This installs the package in editable mode and registers three CLI commands:
`train-vae`, `train-field-predictor`, `run-inversion`.

## Data

Place the simulation HDF5 under `data/`:

```
data/nozzle_dataset_temporal_n1000_min40.h5
```

Expected layout: `/simulations/sim_XXXX/snapshots/t_X.XX/{points, f, u_x, u_y}`,
plus top-level `/curves` (S, M, 2) and `/x_canonical` (M,) arrays. The dataset
itself is not redistributed in this repository.

## Training

Stage-1 (FNO-VAE):

```bash
bash scripts/train_vae.sh
```

Stage-2 (field predictor — needs a Stage-1 checkpoint and matching split):

```bash
# Edit configs/field_predictor.yaml: set input.vae_checkpoint and input.split_info.
bash scripts/train_field_predictor.sh
```

## Evaluation

Geometry-targeted latent inversion (Section 3.3 of the paper):

```bash
# Edit configs/inversion.yaml: set checkpoints.vae, .split_info, .field_predictor.
bash scripts/run_inversion.sh
```

Outputs `summary.json` with oracle / argmin / NN-in-geometry metrics.

## Layout

```
src/spray_surrogate/
  layers.py            shared building blocks (SIREN, FiLM, Fourier features)
  encoder.py           FNO encoder
  decoder.py           time-conditioned SIREN decoder
  geometry_head.py     1D INR for nozzle walls
  vae.py               Stage-1 wrapper, VAE loss, beta scheduler
  field_predictor.py   Stage-2 U-Net
  dataset.py           HDF5 dataset with KDE projection cache
  inversion.py         geometry-targeted argmin
  cli.py               train-vae / train-field-predictor / run-inversion
configs/               YAML configs (one per entrypoint)
scripts/               bash wrappers around the CLI entrypoints
tests/                 pytest shape and convergence sanity checks
```

## Tests

```bash
pytest
```

The tests cover forward-pass tensor shapes, that the inversion procedure
reduces its objective, and that the dataset loader handles a small synthetic
HDF5. They run on CPU in seconds.

## Reproducibility

Each run writes to `experiments/<task>/run_<timestamp>/` containing:

- `config.yaml` (the exact configuration used)
- `git_commit.txt` (current commit hash, when run from a git checkout)
- `best.pt`, `final.pt` (checkpoints)
- `split_info.pt` (train/val/test simulation indices)
- `tb/` (TensorBoard logs, VAE training only)

Configurations (seed, hyperparameters, paths) are fully captured by the YAML
file. Running the same config produces a reproducible result up to the
non-determinism from CUDA reductions.
