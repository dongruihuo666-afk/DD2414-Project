#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simplebev}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${HOME}/datasets/nuscenes}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export MPLCONFIGDIR TORCH_HOME
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python train_nuscenes.py \
    --exp_name="${EXP_NAME:-dd2414_mini_smoke}" \
    --max_iters="${MAX_ITERS:-5}" \
    --log_freq=1000 \
    --dset=mini \
    --do_val=False \
    --save_freq="${SAVE_FREQ:-5}" \
    --batch_size=1 \
    --grad_acc=1 \
    --use_scheduler=False \
    --nworkers="${NUM_WORKERS:-0}" \
    --data_dir="${NUSCENES_ROOT}" \
    --log_dir="${LOG_DIR:-${REPO_ROOT}/logs_nuscenes}" \
    --ckpt_dir="${CKPT_DIR:-${REPO_ROOT}/checkpoints}" \
    --res_scale="${RES_SCALE:-0.5}" \
    --rand_flip=False \
    --rand_crop_and_resize=False \
    --ncams=6 \
    --nsweeps="${NSWEEPS:-1}" \
    --encoder_type="${ENCODER_TYPE:-effb0}" \
    --do_rgbcompress=True \
    --do_shuffle_cams=False \
    --device_ids='[0]' \
    "$@"
