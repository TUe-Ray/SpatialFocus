#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_SCRIPT="${BASE_SCRIPT:-$PROJECT_ROOT/eval_cut3r_sel.sh}"
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/eomt_cut3r_selective_4modes_r2}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
JOB_PREFIX="${JOB_PREFIX:-eval_cut3r_sel3dr2}"
TRAIN_JOB_PREFIX="${TRAIN_JOB_PREFIX:-cut3r_eomt_sel3dr2}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
LIMIT="${LIMIT:-0}"

# Space-separated list of 4 training job IDs corresponding to the 4 modes (in order):
#   soft_word_match_all3d, soft_word_match_zero3d, soft_no_word_match_all3d, soft_no_word_match_zero3d
# Example:
#   TRAIN_JOB_IDS="40500001 40500002 40500003 40500004" bash submit_eval_cut3r_sel_4modes.sh
# When set, each eval job gets --dependency=afterok:<train_job_id> and the run directory is
# derived automatically as $TRAIN_SAVE_ROOT/${TRAIN_JOB_PREFIX}_<short>_<train_job_id>.
# When unset, populate RUN_DIRS below manually.
TRAIN_JOB_IDS="${TRAIN_JOB_IDS:-}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "[ERROR] Base script not found: $BASE_SCRIPT"
    exit 1
fi

mkdir -p "$EVAL_OUTPUT_ROOT"

MODES=(
    "soft_word_match_all3d"
    "soft_word_match_zero3d"
    "soft_no_word_match_all3d"
    "soft_no_word_match_zero3d"
)

declare -A RUN_DIRS=()
# Populated automatically when TRAIN_JOB_IDS is set; otherwise fill in manually:
#   [soft_word_match_all3d]="cut3r_eomt_sel3dr2_wmall_<job_id>"
#   [soft_word_match_zero3d]="cut3r_eomt_sel3dr2_wmzero_<job_id>"
#   [soft_no_word_match_all3d]="cut3r_eomt_sel3dr2_nowmall_<job_id>"
#   [soft_no_word_match_zero3d]="cut3r_eomt_sel3dr2_nowmzero_<job_id>"

short_mode_name() {
    case "$1" in
        soft_word_match_all3d) echo "wmall" ;;
        soft_word_match_zero3d) echo "wmzero" ;;
        soft_no_word_match_all3d) echo "nowmall" ;;
        soft_no_word_match_zero3d) echo "nowmzero" ;;
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

# Parse TRAIN_JOB_IDS into array (if provided)
declare -a TRAIN_IDS=()
if [[ -n "$TRAIN_JOB_IDS" ]]; then
    read -ra TRAIN_IDS <<< "$TRAIN_JOB_IDS"
    if [[ ${#TRAIN_IDS[@]} -ne ${#MODES[@]} ]]; then
        echo "[ERROR] Expected ${#MODES[@]} training job IDs in TRAIN_JOB_IDS, got ${#TRAIN_IDS[@]}"
        echo "  Mode order: ${MODES[*]}"
        exit 1
    fi
fi

echo "[SUBMIT] base_script=$BASE_SCRIPT"
echo "[SUBMIT] train_save_root=$TRAIN_SAVE_ROOT"
echo "[SUBMIT] eval_output_root=$EVAL_OUTPUT_ROOT"
echo "[SUBMIT] job_prefix=$JOB_PREFIX"
echo "[SUBMIT] train_job_prefix=$TRAIN_JOB_PREFIX"
echo "[SUBMIT] modes=${MODES[*]}"
if [[ ${#TRAIN_IDS[@]} -gt 0 ]]; then
    echo "[SUBMIT] train_job_ids=${TRAIN_IDS[*]}"
fi

for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    short_name="$(short_mode_name "$mode")"

    dependency_args=()
    if [[ ${#TRAIN_IDS[@]} -gt 0 ]]; then
        train_job_id="${TRAIN_IDS[$i]}"
        run_name="${TRAIN_JOB_PREFIX}_${short_name}_${train_job_id}"
        run_dir="$TRAIN_SAVE_ROOT/$run_name"
        dependency_args=(--dependency "afterok:${train_job_id}")
        output_path="$EVAL_OUTPUT_ROOT/$run_name"
        pretrained_local="$run_dir"
    else
        if [[ -z "${RUN_DIRS[$mode]+x}" ]]; then
            echo "[ERROR] No run directory configured for mode=$mode and TRAIN_JOB_IDS not set."
            echo "  Either set TRAIN_JOB_IDS or fill in RUN_DIRS in this script."
            exit 1
        fi
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
        output_path="$EVAL_OUTPUT_ROOT/$run_name"
    fi

    job_name="${JOB_PREFIX}_${short_name}"
    eval_run_name="${run_name}_vsibench"
    note="VSI-Bench eval for CUT3R + EoMT selective mode=${mode} (dep on ${train_job_id:-manual})"

    echo "[SUBMIT] mode=$mode job_name=$job_name run_dir=$run_dir"
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
            "${dependency_args[@]}" \
            --job-name="$job_name" \
            --export=ALL,PRETRAINED_LOCAL,MODEL_BASE_LOCAL,SIGLIP_LOCAL,MODEL_NAME,CONV_TEMPLATE,MAX_FRAMES_NUM,RUN_NAME,OUTPUT_PATH,NOTE,LIMIT \
            "$BASE_SCRIPT"
    )
    echo "[SUBMIT] $submit_output"
done

echo "[DONE] Submitted all ${#MODES[@]} evaluation jobs."

echo "[DONE] Submitted all 4 evaluation jobs."