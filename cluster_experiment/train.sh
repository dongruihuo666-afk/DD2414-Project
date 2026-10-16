#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"

: "${BEVCAR_SOURCE_DIR:?Set BEVCAR_SOURCE_DIR to the pinned BEVCar checkout}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to persistent cluster storage}"
if [[ "${MODE}" != "verify" ]]; then
    : "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes v1.0-trainval root}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/configs/radar_scaling_manifest_seed125.json}"
SEED="${SEED:-125}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
VALIDATION_EPOCHS="${VALIDATION_EPOCHS:-1,3,5,8}"
CHECKPOINT_EVERY_UPDATES="${CHECKPOINT_EVERY_UPDATES:-1000}"
PRINT_EVERY_UPDATES="${PRINT_EVERY_UPDATES:-50}"
ACTIVE_TRAINING_PID=""

forward_signal() {
    local signal_name="$1"
    if [[ -n "${ACTIVE_TRAINING_PID}" ]] \
       && kill -0 "${ACTIVE_TRAINING_PID}" 2>/dev/null; then
        kill "-${signal_name}" "${ACTIVE_TRAINING_PID}" 2>/dev/null || true
    fi
}
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM

PRETRAIN_DIR="${OUTPUT_ROOT}/hybrid_teacher_pretrain_10sweep_seed${SEED}"
FINETUNE_DIR="${OUTPUT_ROOT}/hybrid_teacher_end_to_end_10sweep_seed${SEED}"
PROBE_DIR="${OUTPUT_ROOT}/hybrid_teacher_frozen_1x1_probe_10sweep_seed${SEED}"
REPORT_DIR="${OUTPUT_ROOT}/comparison"
PRETRAIN_CHECKPOINT="${PRETRAIN_DIR}/checkpoints/latest.pt"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

check_environment() {
    [[ -x "$(command -v "${PYTHON_BIN}")" ]] || {
        echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1;
    }
    [[ -d "${NUSCENES_ROOT}/v1.0-trainval" ]] || {
        echo "Missing nuScenes v1.0-trainval metadata under ${NUSCENES_ROOT}" >&2; exit 1;
    }
    [[ -f "${BEVCAR_SOURCE_DIR}/nets/voxelnet.py" ]] || {
        echo "Missing BEVCar nets/voxelnet.py under ${BEVCAR_SOURCE_DIR}" >&2; exit 1;
    }
    [[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}" >&2; exit 1; }
    "${PYTHON_BIN}" cluster_experiment/verify_rpl_env.py \
        --mode gpu --precision "${PRECISION}" \
        --bevcar-source "${BEVCAR_SOURCE_DIR}" \
        --require-dinov2-cache
    "${PYTHON_BIN}" -m py_compile \
        scripts/hybrid_teacher_full.py \
        scripts/train_hybrid_teacher_pretrain.py \
        scripts/train_hybrid_teacher_downstream.py
    echo "Environment check passed."
}

resume_argument() {
    local run_dir="$1"
    if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
        printf '%s' '--resume'
    fi
}

run_logged() {
    local run_dir="$1"
    shift
    mkdir -p "${run_dir}"
    printf 'Command:' | tee -a "${run_dir}/console.log"
    printf ' %q' "$@" | tee -a "${run_dir}/console.log"
    printf '\n' | tee -a "${run_dir}/console.log"
    "$@" > >(tee -a "${run_dir}/console.log") 2>&1 &
    ACTIVE_TRAINING_PID=$!
    local wait_status=0
    while true; do
        set +e
        wait "${ACTIVE_TRAINING_PID}"
        wait_status=$?
        set -e
        if (( wait_status >= 128 )) \
           && kill -0 "${ACTIVE_TRAINING_PID}" 2>/dev/null; then
            continue
        fi
        break
    done
    ACTIVE_TRAINING_PID=""
    return "${wait_status}"
}

run_pretrain() {
    if [[ -f "${PRETRAIN_DIR}/summary.json" ]] \
       && grep -q '"status": "complete"' "${PRETRAIN_DIR}/summary.json"; then
        echo "Hybrid-teacher pretraining is already complete."
        return
    fi
    local resume=()
    [[ -n "$(resume_argument "${PRETRAIN_DIR}")" ]] && resume=(--resume)
    run_logged "${PRETRAIN_DIR}" "${PYTHON_BIN}" scripts/train_hybrid_teacher_pretrain.py \
        --data-root "${NUSCENES_ROOT}" \
        --bevcar-source "${BEVCAR_SOURCE_DIR}" \
        --manifest "${MANIFEST}" \
        --run-dir "${PRETRAIN_DIR}" \
        --scale full --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
        --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
        --num-workers "${NUM_WORKERS}" --precision "${PRECISION}" \
        --seed "${SEED}" --nsweeps 10 \
        --validation-epochs "${VALIDATION_EPOCHS}" \
        --checkpoint-every-updates "${CHECKPOINT_EVERY_UPDATES}" \
        --print-every-updates "${PRINT_EVERY_UPDATES}" "${resume[@]}"
}

require_complete_pretrain() {
    [[ -f "${PRETRAIN_CHECKPOINT}" ]] || {
        echo "Missing pretraining checkpoint: ${PRETRAIN_CHECKPOINT}" >&2; exit 1;
    }
    [[ -f "${PRETRAIN_DIR}/summary.json" ]] \
        && grep -q '"status": "complete"' "${PRETRAIN_DIR}/summary.json" || {
        echo "Pretraining has not completed; downstream stages will not start." >&2; exit 1;
    }
}

run_downstream() {
    local stage_mode="$1"
    local run_dir="$2"
    require_complete_pretrain
    if [[ -f "${run_dir}/summary.json" ]] \
       && grep -q '"status": "complete"' "${run_dir}/summary.json"; then
        echo "${stage_mode} is already complete."
        return
    fi
    local resume=()
    [[ -n "$(resume_argument "${run_dir}")" ]] && resume=(--resume)
    run_logged "${run_dir}" "${PYTHON_BIN}" scripts/train_hybrid_teacher_downstream.py \
        --mode "${stage_mode}" \
        --data-root "${NUSCENES_ROOT}" \
        --bevcar-source "${BEVCAR_SOURCE_DIR}" \
        --manifest "${MANIFEST}" \
        --pretrain-checkpoint "${PRETRAIN_CHECKPOINT}" \
        --run-dir "${run_dir}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
        --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
        --num-workers "${NUM_WORKERS}" --precision "${PRECISION}" \
        --seed "${SEED}" --validation-epochs "${VALIDATION_EPOCHS}" \
        --checkpoint-every-updates "${CHECKPOINT_EVERY_UPDATES}" \
        --print-every-updates "${PRINT_EVERY_UPDATES}" "${resume[@]}"
}

run_report() {
    "${PYTHON_BIN}" scripts/summarize_hybrid_teacher_cluster.py \
        --finetune-summary "${FINETUNE_DIR}/summary.json" \
        --linear-probe-summary "${PROBE_DIR}/summary.json" \
        --output-dir "${REPORT_DIR}"
}

run_smoke() {
    local smoke_root="${SMOKE_ROOT:-${OUTPUT_ROOT}/smoke_$(date +%Y%m%d_%H%M%S)}"
    local smoke_pretrain="${smoke_root}/pretrain"
    local smoke_checkpoint="${smoke_pretrain}/checkpoints/latest.pt"
    [[ ! -e "${smoke_root}" ]] || {
        echo "Smoke output already exists: ${smoke_root}" >&2; exit 1;
    }
    run_logged "${smoke_pretrain}" "${PYTHON_BIN}" scripts/train_hybrid_teacher_pretrain.py \
        --data-root "${NUSCENES_ROOT}" --bevcar-source "${BEVCAR_SOURCE_DIR}" \
        --manifest "${MANIFEST}" --run-dir "${smoke_pretrain}" \
        --scale full --train-samples 4 --val-samples 4 --epochs 1 \
        --batch-size 1 --num-workers 0 --precision "${PRECISION}" \
        --seed "${SEED}" --nsweeps 10 \
        --validation-epochs 1 --checkpoint-every-updates 1 --print-every-updates 1
    for stage_mode in finetune linear-probe; do
        run_logged "${smoke_root}/${stage_mode}" \
            "${PYTHON_BIN}" scripts/train_hybrid_teacher_downstream.py \
            --mode "${stage_mode}" --data-root "${NUSCENES_ROOT}" \
            --bevcar-source "${BEVCAR_SOURCE_DIR}" --manifest "${MANIFEST}" \
            --pretrain-checkpoint "${smoke_checkpoint}" \
            --run-dir "${smoke_root}/${stage_mode}" --train-samples 4 \
            --val-samples 4 --epochs 1 --batch-size 1 --num-workers 0 \
            --precision "${PRECISION}" --seed "${SEED}" \
            --validation-epochs 1 \
            --checkpoint-every-updates 1 --print-every-updates 1
    done
    "${PYTHON_BIN}" scripts/summarize_hybrid_teacher_cluster.py \
        --finetune-summary "${smoke_root}/finetune/summary.json" \
        --linear-probe-summary "${smoke_root}/linear-probe/summary.json" \
        --output-dir "${smoke_root}/comparison"
    echo "Smoke protocol passed. Remove ${smoke_root} when it is no longer needed."
}

verify_gpu_environment() {
    "${PYTHON_BIN}" cluster_experiment/verify_rpl_env.py \
        --mode gpu --precision "${PRECISION}" \
        --bevcar-source "${BEVCAR_SOURCE_DIR}" \
        --require-dinov2-cache
}

case "${MODE}" in
    verify) verify_gpu_environment ;;
    check) check_environment ;;
    smoke) check_environment; run_smoke ;;
    pretrain) check_environment; run_pretrain ;;
    finetune) check_environment; run_downstream finetune "${FINETUNE_DIR}" ;;
    linear-probe) check_environment; run_downstream linear-probe "${PROBE_DIR}" ;;
    evaluate) run_report ;;
    all)
        check_environment
        run_pretrain
        run_downstream finetune "${FINETUNE_DIR}"
        run_downstream linear-probe "${PROBE_DIR}"
        run_report
        ;;
    *)
        echo "Usage: $0 {verify|check|smoke|pretrain|finetune|linear-probe|evaluate|all}" >&2
        exit 2
        ;;
esac
