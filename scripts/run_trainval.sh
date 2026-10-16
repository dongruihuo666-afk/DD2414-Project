#!/usr/bin/env bash
# Preview by default. --execute is required to load data or run a model.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/configs/trainval.env"
export CONDA_ROOT CONDA_ENV NUSCENES_ROOT BEVCAR_SOURCE_DIR TORCH_HOME MPLCONFIGDIR
export DSET NSWEEPS NUM_WORKERS TRAIN_SAMPLES VAL_SAMPLES STEPS SEED_LIST
export SAMPLE_SELECTION TAG OUTPUT_DIR

mode="${1:-dual-teacher}"
if (($#)); then shift; fi
execute=false
extra=()
for arg in "$@"; do
    case "$arg" in
        --execute) execute=true ;;
        *) extra+=("$arg") ;;
    esac
done
python_bin="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
common=(--data-root "${NUSCENES_ROOT}" --dset trainval --output-dir "${OUTPUT_DIR}")
probe=(--train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}"
       --steps "${STEPS}" --seed-list "${SEED_LIST}" --nsweeps "${NSWEEPS}"
       --sample-selection "${SAMPLE_SELECTION}" --tag "${TAG}")
case "$mode" in
    data-check)
        command=("${python_bin}" scripts/smoke_test_mini.py "${common[@]}"
                 --num-workers 0 --nsweeps "${NSWEEPS}" --data-only) ;;
    smoke)
        command=("${python_bin}" scripts/smoke_test_mini.py "${common[@]}"
                 --num-workers 0 --nsweeps "${NSWEEPS}" --encoder-type res101
                 --use-radar --use-metaradar --no-pretrained-backbone
                 --visualization-name trainval_radar_smoke.png) ;;
    dual-teacher)
        command=("${python_bin}" scripts/eval_image_distill_heldout.py
                 "${common[@]}" "${probe[@]}") ;;
    motion)
        command=("${python_bin}" scripts/compare_motion_target_bevcar_heldout.py
                 "${common[@]}" "${probe[@]}" --bevcar-source "${BEVCAR_SOURCE_DIR}") ;;
    supervised)
        command=("${python_bin}" train_nuscenes.py --dset=trainval
                 "--data_dir=${NUSCENES_ROOT}" "--exp_name=${EXP_NAME}"
                 "--max_iters=${MAX_ITERS}" "--batch_size=${BATCH_SIZE}"
                 "--grad_acc=${GRAD_ACC}" --nworkers=0 --nworkers_val=0
                 "--res_scale=${RES_SCALE}" "--nsweeps=${NSWEEPS}"
                 --encoder_type=res101 --use_radar=True --use_metaradar=True
                 --use_radar_filters=False --do_rgbcompress=True --ncams=6
                 --do_val=True --val_freq=100 --log_freq=1000
                 "--save_freq=${SAVE_FREQ}" "--use_scheduler=${USE_SCHEDULER}"
                 "--log_dir=${LOG_DIR}" "--ckpt_dir=${CKPT_DIR}" '--device_ids=[0]') ;;
    eval)
        if [[ -z "${INIT_DIR:-}" ]]; then
            echo 'Set INIT_DIR to the matching supervised checkpoint directory.' >&2
            exit 2
        fi
        command=("${python_bin}" eval_nuscenes.py --dset=trainval
                 "--data_dir=${NUSCENES_ROOT}" "--init_dir=${INIT_DIR}"
                 --batch_size=1 --nworkers=0 "--res_scale=${RES_SCALE}"
                 "--nsweeps=${NSWEEPS}" --encoder_type=res101
                 --use_radar=True --use_metaradar=True --use_radar_filters=False
                 --do_rgbcompress=True "--log_dir=${LOG_DIR}/eval" '--device_ids=[0]') ;;
    *)
        echo 'Usage: bash scripts/run_trainval.sh {data-check|smoke|dual-teacher|motion|supervised|eval} [--execute] [extra arguments]' >&2
        exit 2 ;;
esac
command+=("${extra[@]}")
printf 'Working directory: %s\n' "${REPO_ROOT}"
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'
if ! "$execute"; then
    echo 'Preview only: no data loader, model, or training was started. Add --execute after reviewing TRAINVAL_RUN_PLAN.md.'
    exit 0
fi
[[ -x "${python_bin}" ]] || { echo "Missing environment Python: ${python_bin}" >&2; exit 1; }
[[ -d "${NUSCENES_ROOT}/v1.0-trainval" ]] || { echo 'Missing v1.0-trainval metadata.' >&2; exit 1; }
if [[ "$mode" == motion && ! -f "${BEVCAR_SOURCE_DIR}/nets/voxelnet.py" ]]; then
    echo "Missing BEVCar source: ${BEVCAR_SOURCE_DIR}; see TRAINVAL_RUN_PLAN.md." >&2
    exit 1
fi
# Activation supplies environment libraries even when interactive conda is not initialized.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}" "${TORCH_HOME}"
cd "${REPO_ROOT}"
exec "${command[@]}"
