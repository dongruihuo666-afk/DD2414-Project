#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"

: "${SLURM_PARTITION:?Set SLURM_PARTITION after checking the cluster documentation}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to persistent cluster storage}"
: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the staged trainval dataset}"
: "${BEVCAR_SOURCE_DIR:?Set BEVCAR_SOURCE_DIR to the pinned BEVCar checkout}"

mkdir -p "${OUTPUT_ROOT}/slurm_logs"
account_args=()
[[ -n "${SLURM_ACCOUNT:-}" ]] && account_args=(--account "${SLURM_ACCOUNT}")

sbatch \
    --job-name="dd2414-hybrid-${MODE}" \
    --partition="${SLURM_PARTITION}" \
    "${account_args[@]}" \
    --gres="gpu:${SLURM_GPUS:-1}" \
    --cpus-per-task="${SLURM_CPUS:-10}" \
    --mem="${SLURM_MEMORY:-64G}" \
    --time="${SLURM_TIME:-48:00:00}" \
    --signal=B:USR1@180 \
    --output="${OUTPUT_ROOT}/slurm_logs/%x-%j.log" \
    --export=ALL \
    --wrap="cd '${REPO_ROOT}' && exec bash cluster_experiment/train.sh '${MODE}'"
