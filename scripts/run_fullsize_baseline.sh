#!/usr/bin/env bash
# Preview by default. --execute is required to start training.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/configs/trainval.env"

scale="${SCALE:-full}"
epochs="${EPOCHS:-8}"
batch_size="${FULLSIZE_BATCH_SIZE:-5}"
precision="${PRECISION:-bf16}"
num_workers="${FULLSIZE_NUM_WORKERS:-8}"
seed="${SEED:-125}"
case "${scale}" in
    4096) run_name="scale4096_seed${seed}" ;;
    full) run_name="full28130_seed${seed}" ;;
    *) echo 'SCALE must be 4096 or full.' >&2; exit 2 ;;
esac
run_dir="${RUN_DIR:-${REPO_ROOT}/fullsize_baseline/runs/${run_name}}"
python_bin="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
execute=false
extra=()
for arg in "$@"; do
    case "$arg" in
        --execute) execute=true ;;
        *) extra+=("$arg") ;;
    esac
done
command=(
    "${python_bin}" scripts/train_fullsize_baseline.py
    --data-root "${NUSCENES_ROOT}"
    --bevcar-source "${BEVCAR_SOURCE_DIR}"
    --manifest "${REPO_ROOT}/configs/radar_scaling_manifest_seed125.json"
    --scale "${scale}"
    --run-dir "${run_dir}"
    --epochs "${epochs}"
    --batch-size "${batch_size}"
    --precision "${precision}"
    --num-workers "${num_workers}"
    --seed "${seed}"
    --nsweeps 1
    "${extra[@]}"
)
printf 'Working directory: %s\n' "${REPO_ROOT}"
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'
if ! "${execute}"; then
    echo 'Preview only. Add --execute after the batch/precision pilot is reviewed.'
    exit 0
fi
[[ -x "${python_bin}" ]] || { echo "Missing environment Python: ${python_bin}" >&2; exit 1; }
[[ -d "${NUSCENES_ROOT}/v1.0-trainval" ]] || { echo 'Missing v1.0-trainval metadata.' >&2; exit 1; }
[[ -f "${BEVCAR_SOURCE_DIR}/nets/voxelnet.py" ]] || { echo 'Missing pinned BEVCar VoxelNet.' >&2; exit 1; }
export TORCH_HOME MPLCONFIGDIR
mkdir -p "${run_dir}" "${MPLCONFIGDIR}" "${TORCH_HOME}"
cd "${REPO_ROOT}"
"${command[@]}" 2>&1 | tee -a "${run_dir}/console.log"
