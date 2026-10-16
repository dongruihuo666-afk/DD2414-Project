#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export MPLCONFIGDIR

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/smoke_test_mini.py \
    --data-root "${NUSCENES_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --encoder-type effb0 \
    --res-scale 0.5 \
    --nsweeps 1 \
    --use-radar-encoder \
    --no-pretrained-backbone \
    --visualization-name radar_encoder_fusion_smoke.png \
    "$@"
