#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"
BEVCAR_SOURCE_DIR="${BEVCAR_SOURCE_DIR:-${REPO_ROOT}/external/BEVCar}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export MPLCONFIGDIR

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/compare_radar_velocity_tiny.py \
    --data-root "${NUSCENES_ROOT}" \
    --bevcar-source "${BEVCAR_SOURCE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
