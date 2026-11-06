#!/usr/bin/env bash
# Submitted explicitly later: no SSH, sbatch, queue, requeue, or job cancellation.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm allocation required' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
TREATMENT="$2"
shift 2
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/Midgard/home/hongen/miniforge3/bin/python}"
ENV_PREFIX="$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["env_prefix"])' "${CONFIG}")"
cd "${ROOT}"
# The wrapper remains the batch PID for --signal=B:USR1@300. It writes a stop
# flag; do not forward this signal to torchrun or directly kill its children.
exec "${ENV_PREFIX}/bin/python" "${ROOT}/cluster_rpl/depth_study.py" \
  --config "${CONFIG}" --treatment "${TREATMENT}" --execute "$@"
