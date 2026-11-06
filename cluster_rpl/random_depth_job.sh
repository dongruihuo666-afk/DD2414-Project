#!/usr/bin/env bash
# Explicit sbatch entry; Slurm executes a spool copy of this file.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm allocation required' >&2; exit 2; }
ROOT="${DD2414_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT="$(cd "${ROOT}" && pwd -P)"
[[ -f "${ROOT}/cluster_rpl/random_depth_study.py" ]] || { echo 'Explicit project root required' >&2; exit 2; }
CONFIG="$1"
TREATMENT="$2"
shift 2
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/Midgard/home/hongen/miniforge3/bin/python}"
ENV_PREFIX="$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["env_prefix"])' "${CONFIG}")"
cd "${ROOT}"
exec "${ENV_PREFIX}/bin/python" "${ROOT}/cluster_rpl/random_depth_study.py" \
  --config "${CONFIG}" --treatment "${TREATMENT}" --execute "$@"
