#!/usr/bin/env bash
# Launch the approved baseline study as a persistent local user service.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/bev/bin/python}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
BEVCAR_SOURCE_DIR="${BEVCAR_SOURCE_DIR:-${REPO_ROOT}/external/BEVCar}"
STUDY_DIR="${STUDY_DIR:-${REPO_ROOT}/fullsize_baseline/runs/meeting_followup_20261016}"
RUN_SERVICE="${RUN_SERVICE:-dd2414-baseline-followup-day}"
cd "${REPO_ROOT}"
[[ "$(git branch --show-current)" == dd2414-mini-baseline ]] || {
    echo 'Switch to dd2414-mini-baseline before starting this study.' >&2; exit 1;
}
[[ -d "${NUSCENES_ROOT}/v1.0-trainval" ]] || { echo 'Missing trainval dataset.' >&2; exit 1; }
[[ -f "${BEVCAR_SOURCE_DIR}/nets/voxelnet.py" ]] || { echo 'Missing BEVCar VoxelNet.' >&2; exit 1; }
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))'
systemctl --user show-environment >/dev/null
if systemctl --user is-active --quiet "${RUN_SERVICE}.service"; then
    echo "${RUN_SERVICE} is already active."; exit 0
fi
mkdir -p "${STUDY_DIR}"
systemd-run --user --unit="${RUN_SERVICE}" \
    --working-directory="${REPO_ROOT}" \
    --description='DD2414 meeting follow-up full-data sequential GPU queue' \
    --property=Restart=on-failure --property=RestartSec=60 \
    --property=StartLimitIntervalSec=86400 --property=StartLimitBurst=3 \
    --property=TimeoutStopSec=180 --property=RuntimeMaxSec=90000 \
    --property="StandardOutput=append:${STUDY_DIR}/queue.log" \
    --property=StandardError=inherit \
    /usr/bin/env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    TORCH_HOME="${REPO_ROOT}/.cache/torch" MPLCONFIGDIR="${REPO_ROOT}/.cache/matplotlib" \
    SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    "${PYTHON_BIN}" scripts/run_baseline_followup_day.py \
    --study-dir "${STUDY_DIR}" --hours 24 --batch-size "${BATCH_SIZE:-5}" \
    --num-workers "${NUM_WORKERS:-8}" --data-root "${NUSCENES_ROOT}" \
    --bevcar-source "${BEVCAR_SOURCE_DIR}"
echo "Logs: ${STUDY_DIR}/queue.log"
echo "Monitor: ${PYTHON_BIN} scripts/monitor_baseline_followup.py --study-dir ${STUDY_DIR}"
