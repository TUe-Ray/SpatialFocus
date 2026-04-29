#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_SCRIPT="${BASE_SCRIPT:-$PROJECT_ROOT/eval_cut3r_sel.sh}"
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/eomt_cut3r_selective_4modes}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
JOB_PREFIX="${JOB_PREFIX:-eval_cut3r_sel3d}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
LIMIT="${LIMIT:-0}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "[ERROR] Base script not found: $BASE_SCRIPT"
    exit 1
fi

mkdir -p "$EVAL_OUTPUT_ROOT"

MODES=(
    "baseline"
    "selective_soft"
    "selective_soft_with_floor"
    "selective_soft_with_floor_zero_fallback"
)

declare -A RUN_DIRS=(
    [baseline]="cut3r_eomt_sel3d_base_40301264"
    [selective_soft]="cut3r_eomt_sel3d_soft_40301265"
    [selective_soft_with_floor]="cut3r_eomt_sel3d_softfloor_40301266"
    [selective_soft_with_floor_zero_fallback]="cut3r_eomt_sel3d_softfloor0_40301267"
)

short_mode_name() {
    case "$1" in
        baseline) echo "base" ;;
        selective_soft) echo "soft" ;;
        selective_soft_with_floor) echo "softfloor" ;;
        selective_soft_with_floor_zero_fallback) echo "softfloor0" ;;
        *) echo "unknown" ;;
    esac
}

find_latest_eval_checkpoint() {
    local run_dir="$1"
    local checkpoint_dir
    local -a checkpoint_dirs=()

    mapfile -t checkpoint_dirs < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V -r)

    for checkpoint_dir in "${checkpoint_dirs[@]}"; do
        if [[ -f "$checkpoint_dir/config.json" && -f "$checkpoint_dir/adapter_model.bin" ]]; then
            echo "$checkpoint_dir"
            return 0
        fi
    done

    return 1
}

resolve_eval_model_dir() {
    local run_dir="$1"

    if [[ -f "$run_dir/config.json" && -f "$run_dir/adapter_model.bin" ]]; then
        echo "$run_dir"
        return 0
    fi

    find_latest_eval_checkpoint "$run_dir"
}

echo "[SUBMIT] base_script=$BASE_SCRIPT"
echo "[SUBMIT] train_save_root=$TRAIN_SAVE_ROOT"
echo "[SUBMIT] eval_output_root=$EVAL_OUTPUT_ROOT"
echo "[SUBMIT] job_prefix=$JOB_PREFIX"
echo "[SUBMIT] modes=${MODES[*]}"

for mode in "${MODES[@]}"; do
    run_name="${RUN_DIRS[$mode]}"
    run_dir="$TRAIN_SAVE_ROOT/$run_name"
    if [[ ! -d "$run_dir" ]]; then
        echo "[ERROR] Run directory not found for mode=$mode: $run_dir"
        exit 1
    fi

    if ! pretrained_local="$(resolve_eval_model_dir "$run_dir")"; then
        echo "[ERROR] No evaluable model directory found for mode=$mode in $run_dir"
        exit 1
    fi

    short_name="$(short_mode_name "$mode")"
    job_name="${JOB_PREFIX}_${short_name}"
    eval_run_name="${run_name}_vsibench"
    output_path="$EVAL_OUTPUT_ROOT/$run_name"
    note="VSI-Bench eval for CUT3R + EoMT selective mode=${mode} using $(basename "$pretrained_local")"

    echo "[SUBMIT] mode=$mode job_name=$job_name pretrained_local=$pretrained_local"
    submit_output=$(
        PRETRAINED_LOCAL="$pretrained_local" \
        MODEL_BASE_LOCAL="$MODEL_BASE_LOCAL" \
        SIGLIP_LOCAL="$SIGLIP_LOCAL" \
        MODEL_NAME="$MODEL_NAME" \
        CONV_TEMPLATE="$CONV_TEMPLATE" \
        MAX_FRAMES_NUM="$MAX_FRAMES_NUM" \
        RUN_NAME="$eval_run_name" \
        OUTPUT_PATH="$output_path" \
        NOTE="$note" \
        LIMIT="$LIMIT" \
        sbatch \
            --job-name="$job_name" \
            --export=ALL,PRETRAINED_LOCAL,MODEL_BASE_LOCAL,SIGLIP_LOCAL,MODEL_NAME,CONV_TEMPLATE,MAX_FRAMES_NUM,RUN_NAME,OUTPUT_PATH,NOTE,LIMIT \
            "$BASE_SCRIPT"
    )
    echo "[SUBMIT] $submit_output"
done

echo "[DONE] Submitted all 4 evaluation jobs."