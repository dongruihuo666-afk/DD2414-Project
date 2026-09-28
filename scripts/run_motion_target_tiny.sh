#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export MPLCONFIGDIR TORCH_HOME
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}" "${OUTPUT_DIR}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/compare_motion_target_tiny.py \
    --data-root "${NUSCENES_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples "${SAMPLES:-4}" \
    --steps "${STEPS:-100}" \
    "$@"
