#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export MPLCONFIGDIR TORCH_HOME
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/smoke_test_mini.py \
    --data-root "${NUSCENES_ROOT}" \
    --output-dir "${OUTPUT_DIR:-${REPO_ROOT}/artifacts}" \
    --encoder-type "${ENCODER_TYPE:-effb0}" \
    --res-scale "${RES_SCALE:-0.5}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --nsweeps "${NSWEEPS:-1}" \
    "$@"
