#!/usr/bin/env bash
# Print one environment-creation command; execute only after explicit approval.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_BIN="${MAMBA_BIN:-/Midgard/home/hongen/miniforge3/bin/mamba}"
ENV_PREFIX="${ENV_PREFIX:-/Midgard/home/hongen/miniforge3/envs/dd2414}"
[[ $# -le 1 && ( $# -eq 0 || "$1" == --execute ) ]] || {
    echo 'Usage: bash cluster_rpl/setup_environment.sh [--execute]' >&2; exit 2;
}
command=("${MAMBA_BIN}" env create --prefix "${ENV_PREFIX}" --file "${ROOT}/cluster_rpl/environment.yml")
printf '%q ' "${command[@]}"
printf '\n'
if [[ $# -eq 1 ]]; then
    [[ -x "${MAMBA_BIN}" ]] || { echo 'mamba is missing' >&2; exit 1; }
    [[ ! -e "${ENV_PREFIX}" ]] || {
        echo 'Environment path already exists; inspect it, do not overwrite.' >&2; exit 1;
    }
    exec "${command[@]}"
fi
