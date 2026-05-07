#!/usr/bin/env bash
set -euo pipefail
train-vae --config configs/vae.yaml "$@"
