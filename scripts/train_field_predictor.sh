#!/usr/bin/env bash
set -euo pipefail
train-field-predictor --config configs/field_predictor.yaml "$@"
