#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_SCRIPT="${BASE_SCRIPT:-$PROJECT_ROOT/eval_cut3r_obj.sh}"
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/eomt_cut3r_obj_4modes}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
JOB_PREFIX="${JOB_PREFIX:-eval_cut3r_eomt_obj}"
TRAIN_JOB_PREFIX="${TRAIN_JOB_PREFIX:-cut3r_eomt_obj}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
LIMIT="${LIMIT:-0}"

# Space-separated list of 4 training job IDs corresponding to the 4 modes (in order):
#   eomt_obj_only, eomt_obj_text_phrase, eomt_obj_learnable, eomt_obj_only_word_filter
# Example:
#   TRAIN_JOB_IDS="40402248 40402249 40402251 40402253" bash submit_eval_cut3r_obj_5modes.sh
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
    "eomt_obj_only"
    "eomt_obj_text_phrase"
    "eomt_obj_learnable"
    "eomt_obj_only_word_filter"
)

short_mode_name() {
    case "$1" in
        eomt_obj_only) echo "obj" ;;
        eomt_obj_text_phrase) echo "objtext" ;;
        eomt_obj_learnable) echo "objlearn" ;;
        eomt_obj_only_word_filter) echo "objword" ;;
        *) echo "unknown" ;;
    esac
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
echo "[SUBMIT] modes=${MODES[*]}"
if [[ ${#TRAIN_IDS[@]} -gt 0 ]]; then
    echo "[SUBMIT] train_job_ids=${TRAIN_IDS[*]}"
fi

for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    short_name="$(short_mode_name "$mode")"
    job_name="${JOB_PREFIX}_${short_name}"

    dependency_args=()
    if [[ ${#TRAIN_IDS[@]} -gt 0 ]]; then
        train_job_id="${TRAIN_IDS[$i]}"
        run_dir="$TRAIN_SAVE_ROOT/${TRAIN_JOB_PREFIX}_${short_name}_${train_job_id}"
        dependency_args=(--dependency "afterok:${train_job_id}")
        output_path="$EVAL_OUTPUT_ROOT/${TRAIN_JOB_PREFIX}_${short_name}_${train_job_id}"
    else
        echo "[ERROR] TRAIN_JOB_IDS is not set. Provide it as a space-separated list of ${#MODES[@]} job IDs."
        exit 1
    fi

    eval_run_name="${TRAIN_JOB_PREFIX}_${short_name}_vsibench"
    note="VSI-Bench eval for CUT3R + EoMT obj mode=${mode} (dep on ${train_job_id})"

    echo "[SUBMIT] mode=$mode job_name=$job_name run_dir=$run_dir"
    submit_output=$(
        PRETRAINED_LOCAL="$run_dir" \
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
