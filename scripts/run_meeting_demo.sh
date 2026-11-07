#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RGB_CKPT="${RGB_CKPT:-${REPO_ROOT}/checkpoints/8x5_5e-4_rgb12_22:43:46}"
RAD_CKPT="${RAD_CKPT:-${REPO_ROOT}/checkpoints/8x5_5e-4_rad25_18:55:34}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"

if [[ ! -d "${RGB_CKPT}" || ! -d "${RAD_CKPT}" ]]; then
    echo "Official checkpoints are missing. See SETUP_LOCAL.md." >&2
    exit 1
fi

"${REPO_ROOT}/scripts/smoke_test_mini.sh" \
    --encoder-type res101 \
    --checkpoint "${RGB_CKPT}" \
    --no-backward \
    --visualization-name mini_camera_pretrained.png \
    --dump-npz "${OUTPUT_DIR}/meeting_camera.npz"

"${REPO_ROOT}/scripts/smoke_test_mini.sh" \
    --encoder-type res101 \
    --checkpoint "${RAD_CKPT}" \
    --use-radar --use-metaradar \
    --no-backward \
    --visualization-name mini_radar_pretrained.png \
    --dump-npz "${OUTPUT_DIR}/meeting_radar.npz"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python "${REPO_ROOT}/scripts/make_meeting_comparison.py" \
    --camera "${OUTPUT_DIR}/meeting_camera.npz" \
    --radar "${OUTPUT_DIR}/meeting_radar.npz" \
    --output "${OUTPUT_DIR}/meeting_demo_comparison.png"

echo "Meeting demo ready: ${OUTPUT_DIR}/meeting_demo_comparison.png"
