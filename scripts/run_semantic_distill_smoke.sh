#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
RAD_CKPT="${RAD_CKPT:-${REPO_ROOT}/checkpoints/8x5_5e-4_rad25_18:55:34}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export TORCH_HOME

if [[ ! -d "${RAD_CKPT}" ]]; then
    echo "Official radar checkpoint is missing: ${RAD_CKPT}" >&2
    exit 1
fi
if [[ ! -f "${OUTPUT_DIR}/dinov2_radar_soft_targets.npz" ]]; then
    echo "DINOv2 target cache is missing; run ./scripts/run_dinov2_bev_demo.sh first." >&2
    exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/semantic_distill_smoke.py \
    --data-root "${NUSCENES_ROOT}" \
    --checkpoint "${RAD_CKPT}" \
    --target-cache "${OUTPUT_DIR}/dinov2_radar_soft_targets.npz" \
    --teacher-cache "${OUTPUT_DIR}/dinov2_teacher_features.npz" \
    --output-dir "${OUTPUT_DIR}" \
    --steps "${STEPS:-20}" \
    "$@"
