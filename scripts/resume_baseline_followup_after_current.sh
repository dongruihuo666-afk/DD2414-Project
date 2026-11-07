#!/usr/bin/env bash
# Wait for the current queue without signalling or interrupting its GPU child.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMARY_UNIT="${PRIMARY_UNIT:-dd2414-baseline-followup-day.service}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/bev/bin/python}"
STUDY_DIR="${STUDY_DIR:-${REPO_ROOT}/fullsize_baseline/runs/meeting_followup_20261016}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
BEVCAR_SOURCE_DIR="${BEVCAR_SOURCE_DIR:-${REPO_ROOT}/external/BEVCar}"
cd "${REPO_ROOT}"
echo "Waiting for ${PRIMARY_UNIT}; its current training will not be interrupted."
while true; do
    state="$(systemctl --user show "${PRIMARY_UNIT}" --property=ActiveState --value)"
    case "${state}" in
        active|activating|deactivating|reloading) sleep 30 ;;
        *) break ;;
    esac
done
[[ "$(git branch --show-current)" == dd2414-mini-baseline ]] || {
    echo 'Recovery is restricted to dd2414-mini-baseline.' >&2; exit 1;
}
echo 'Resuming the original study with its existing absolute deadline and completed stages.'
exec "${PYTHON_BIN}" scripts/run_baseline_followup_day.py \
    --study-dir "${STUDY_DIR}" --data-root "${NUSCENES_ROOT}" \
    --bevcar-source "${BEVCAR_SOURCE_DIR}"
