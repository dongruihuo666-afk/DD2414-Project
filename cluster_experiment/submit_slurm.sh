#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"

: "${SLURM_PARTITION:?Set SLURM_PARTITION after checking the cluster documentation}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to persistent cluster storage}"
: "${BEVCAR_SOURCE_DIR:?Set BEVCAR_SOURCE_DIR to the pinned BEVCar checkout}"
if [[ "${MODE}" != "verify" ]]; then
    : "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the staged trainval dataset}"
fi

SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"
PYTHON_BIN="${PYTHON_BIN:-/Midgard/home/hongen/miniforge3/envs/dd2414-hybrid/bin/python}"
TORCH_HOME="${TORCH_HOME:-/Midgard/home/hongen/.cache/torch}"
PRECISION="${PRECISION:-bf16}"

if [[ "${SLURM_GPUS}" != "1" ]]; then
    echo "This experiment is limited to exactly one GPU; got SLURM_GPUS=${SLURM_GPUS}." >&2
    exit 2
fi
if [[ ! "${SLURM_TIME}" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]]; then
    echo "SLURM_TIME must use HH:MM:SS, for example 24:00:00." >&2
    exit 2
fi
hours="${BASH_REMATCH[1]}"
minutes="${BASH_REMATCH[2]}"
seconds="${BASH_REMATCH[3]}"
if (( 10#${hours} > 48 )) \
   || (( 10#${hours} == 48 && (10#${minutes} != 0 || 10#${seconds} != 0) )); then
    echo "SLURM_TIME may not exceed the 48-hour master-student limit." >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}/slurm_logs"
account_args=()
[[ -n "${SLURM_ACCOUNT:-}" ]] && account_args=(--account "${SLURM_ACCOUNT}")
constraint_args=()
[[ -n "${SLURM_CONSTRAINT:-}" ]] && constraint_args=(--constraint "${SLURM_CONSTRAINT}")

export_values="ALL,PYTHON_BIN=${PYTHON_BIN},BEVCAR_SOURCE_DIR=${BEVCAR_SOURCE_DIR},OUTPUT_ROOT=${OUTPUT_ROOT},PRECISION=${PRECISION},TORCH_HOME=${TORCH_HOME}"
if [[ "${MODE}" != "verify" ]]; then
    export_values="${export_values},NUSCENES_ROOT=${NUSCENES_ROOT}"
fi

sbatch \
    --job-name="dd2414-hybrid-${MODE}" \
    --partition="${SLURM_PARTITION}" \
    "${account_args[@]}" \
    "${constraint_args[@]}" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${SLURM_GPUS}" \
    --cpus-per-task="${SLURM_CPUS:-10}" \
    --mem="${SLURM_MEMORY:-64G}" \
    --time="${SLURM_TIME}" \
    --signal=B:USR1@180 \
    --output="${OUTPUT_ROOT}/slurm_logs/%x-%j.log" \
    --export="${export_values}" \
    --wrap="cd '${REPO_ROOT}' && exec bash cluster_experiment/train.sh '${MODE}'"
