#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
RAD_CKPT="${RAD_CKPT:-${REPO_ROOT}/checkpoints/8x5_5e-4_rad25_18:55:34}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"

if [[ ! -d "${RAD_CKPT}" ]]; then
    echo "Official radar checkpoint is missing: ${RAD_CKPT}" >&2
    exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/baseline_overfit_mini.py \
    --data-root "${NUSCENES_ROOT}" \
    --checkpoint "${RAD_CKPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-samples "${NUM_SAMPLES:-4}" \
    --steps "${STEPS:-50}" \
    --nsweeps "${NSWEEPS:-5}" \
    "$@"
