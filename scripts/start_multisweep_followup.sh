#!/usr/bin/env bash
# Baseline-only persistent single-GPU queue, with no wall-time or idle cutoff.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/bev/bin/python}"
STUDY_DIR="${STUDY_DIR:-${REPO_ROOT}/fullsize_baseline/runs/multisweep_teacher_20261016}"
RUN_SERVICE="${RUN_SERVICE:-dd2414-multisweep-teacher}"
cd "${REPO_ROOT}"
[[ "$(git branch --show-current)" == dd2414-mini-baseline ]] || exit 1
for previous in dd2414-baseline-followup-day dd2414-baseline-followup-recovery; do
    if systemctl --user is-active --quiet "${previous}.service"; then
        echo "${previous} is still active; checkpoint/stop the old queue before starting." >&2
        exit 1
    fi
done
if systemctl --user is-active --quiet "${RUN_SERVICE}.service"; then
    echo "${RUN_SERVICE} is already active"; exit 0
fi
[[ -x "${PYTHON_BIN}" ]] || { echo 'Missing environment Python.' >&2; exit 1; }
[[ -d "${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}/v1.0-trainval" ]] || {
    echo 'Missing full nuScenes metadata.' >&2; exit 1;
}
[[ -f "${BEVCAR_SOURCE_DIR:-${REPO_ROOT}/external/BEVCar}/nets/voxelnet.py" ]] || {
    echo 'Missing BEVCar VoxelNet source.' >&2; exit 1;
}
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))'
mkdir -p "${STUDY_DIR}"
systemd-run --user --unit="${RUN_SERVICE}" --working-directory="${REPO_ROOT}" \
    --description='DD2414 baseline double DINO, 5/10 sweeps, depth and GT-object motion' \
    --property=Restart=on-failure --property=RestartSec=60 \
    --property=StartLimitIntervalSec=86400 --property=StartLimitBurst=3 \
    --property=RuntimeMaxSec=infinity --property=TimeoutStopSec=infinity \
    --property=KillMode=mixed \
    --property="StandardOutput=append:${STUDY_DIR}/queue.log" --property=StandardError=inherit \
    /usr/bin/env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_HOME="${REPO_ROOT}/.cache/torch" MPLCONFIGDIR="${REPO_ROOT}/.cache/matplotlib" \
    SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    "${PYTHON_BIN}" scripts/run_multisweep_followup.py --study-dir "${STUDY_DIR}" \
    --data-root "${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}" \
    --bevcar-source "${BEVCAR_SOURCE_DIR:-${REPO_ROOT}/external/BEVCar}" \
    --batch-size "${BATCH_SIZE:-4}" --num-workers "${NUM_WORKERS:-8}" \
    --pretrain-epochs "${PRETRAIN_EPOCHS:-1}" --downstream-epochs "${DOWNSTREAM_EPOCHS:-1}" \
    --legacy-study "${REPO_ROOT}/fullsize_baseline/runs/meeting_followup_20261016"
echo "Progress: ${STUDY_DIR}/pipeline.json"
echo "Queue log: ${STUDY_DIR}/queue.log"
echo "Results: ${STUDY_DIR}/REPORT.md"
