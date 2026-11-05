#!/usr/bin/env bash
# Slurm invokes exactly one task. No background queue, SSH, Git push or requeue.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm allocation required' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
TASK="$2"
shift 2
[[ -f "${CONFIG}" ]] || { echo 'Config missing' >&2; exit 2; }
# Parse paths without importing torch or activating an interactive shell.
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/Midgard/home/hongen/miniforge3/bin/python}"
ENV_PREFIX="$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["env_prefix"])' "${CONFIG}")"
cd "${ROOT}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_HOME="${ROOT}/.cache/torch"
export MPLCONFIGDIR="${ROOT}/.cache/matplotlib"
# exec preserves the batch PID so B:USR1 reaches control.py's checkpoint handler.
exec "${ENV_PREFIX}/bin/python" "${ROOT}/cluster_rpl/control.py" \
    --config "${CONFIG}" run --task "${TASK}" "$@"
