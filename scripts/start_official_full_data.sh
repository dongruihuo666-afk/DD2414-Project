#!/usr/bin/env bash
# Start the approved official-split queue independently of the chat process.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/bev/bin/python}"
STUDY_DIR="${STUDY_DIR:-${REPO_ROOT}/fullsize_baseline/runs/official_rgb_full_10sweep_20261103}"
RUN_SERVICE="dd2414-official-full-data"
cd "${REPO_ROOT}"
[[ "$(git branch --show-current)" == dd2414-mini-baseline ]] || exit 1
[[ -x "${PYTHON_BIN}" ]] || { echo 'Missing bev environment.' >&2; exit 1; }
if systemctl --user is-active --quiet "${RUN_SERVICE}.service"; then
    echo "${RUN_SERVICE} is already active"
    exit 0
fi
mkdir -p "${STUDY_DIR}"
systemd-run --user --unit="${RUN_SERVICE}" --working-directory="${REPO_ROOT}" \
    --description='DD2414 official split, official RGB warm-start, double DINO, ten sweeps' \
    --property=Restart=on-failure --property=RestartSec=60 \
    --property=StartLimitIntervalSec=86400 --property=StartLimitBurst=3 \
    --property=RuntimeMaxSec=infinity --property=TimeoutStopSec=infinity \
    --property=KillMode=mixed \
    --property="StandardOutput=append:${STUDY_DIR}/queue.log" --property=StandardError=inherit \
    /usr/bin/env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_HOME="${REPO_ROOT}/.cache/torch" MPLCONFIGDIR="${REPO_ROOT}/.cache/matplotlib" \
    SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    "${PYTHON_BIN}" scripts/run_official_full_data.py --study-dir "${STUDY_DIR}"
echo "Progress: ${STUDY_DIR}/pipeline.json"
echo "Queue log: ${STUDY_DIR}/queue.log"
