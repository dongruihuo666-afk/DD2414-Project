#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"
BEVCAR_SOURCE_DIR="${BEVCAR_SOURCE_DIR:-${REPO_ROOT}/external/BEVCar}"
CAMERA_CHECKPOINT="${CAMERA_CHECKPOINT:-${REPO_ROOT}/checkpoints/8x5_5e-4_rgb12_22:43:46}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export MPLCONFIGDIR

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/bevcar_supervised_mini.py \
    --data-root "${NUSCENES_ROOT}" \
    --camera-checkpoint "${CAMERA_CHECKPOINT}" \
    --bevcar-source "${BEVCAR_SOURCE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
