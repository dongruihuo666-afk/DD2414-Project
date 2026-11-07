#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
RGB_CKPT="${RGB_CKPT:-${REPO_ROOT}/checkpoints/8x5_5e-4_rgb12_22:43:46}"
RAD_CKPT="${RAD_CKPT:-${REPO_ROOT}/checkpoints/8x5_5e-4_rad25_18:55:34}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"

for checkpoint in "${RGB_CKPT}" "${RAD_CKPT}"; do
    if [[ ! -d "${checkpoint}" ]]; then
        echo "Official checkpoint is missing: ${checkpoint}" >&2
        exit 1
    fi
done

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/eval_mini_subset.py \
    --data-root "${NUSCENES_ROOT}" \
    --camera-checkpoint "${RGB_CKPT}" \
    --radar-checkpoint "${RAD_CKPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-samples "${NUM_SAMPLES:-10}" \
    --nsweeps "${NSWEEPS:-5}" \
    "$@"
