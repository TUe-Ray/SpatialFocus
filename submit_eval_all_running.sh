#!/bin/bash
# Submit eval jobs (with --dependency=afterok) for all currently running training jobs.
# Usage: bash submit_eval_all_running.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/all_running}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
LIMIT="${LIMIT:-0}"

OBJ_EVAL_SCRIPT="${PROJECT_ROOT}/eval_cut3r_obj.sh"
SEL_EVAL_SCRIPT="${PROJECT_ROOT}/eval_cut3r_sel.sh"

for s in "$OBJ_EVAL_SCRIPT" "$SEL_EVAL_SCRIPT"; do
    if [[ ! -f "$s" ]]; then
        echo "[ERROR] Eval script not found: $s"
        exit 1
    fi
done

mkdir -p "$EVAL_OUTPUT_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# Training jobs to evaluate.
# Format per entry: "<train_job_id> <slurm_job_name> <eval_script_key>"
#   eval_script_key: "obj" -> eval_cut3r_obj.sh
#                    "sel" -> eval_cut3r_sel.sh
# ─────────────────────────────────────────────────────────────────────────────
JOBS=(
    "40390735 selec_100%_baseline         sel"
    "40390731 selec_100%_softfloor        sel"
    "40400440 selec_100%_soft             sel"
    "40402248 cut3r_eomt_obj_obj          obj"
    "40402249 cut3r_eomt_obj_objtext      obj"
    "40402251 cut3r_eomt_obj_objlearn     obj"
    "40402253 cut3r_eomt_obj_objword      obj"
    "40403422 eomt_obj_text_phrase_100p   obj"
)

echo "[SUBMIT] train_save_root=$TRAIN_SAVE_ROOT"
echo "[SUBMIT] eval_output_root=$EVAL_OUTPUT_ROOT"
echo "[SUBMIT] jobs=${#JOBS[@]}"
echo ""

for entry in "${JOBS[@]}"; do
    read -r train_job_id train_job_name eval_key <<< "$entry"

    case "$eval_key" in
        obj) eval_script="$OBJ_EVAL_SCRIPT" ;;
        sel) eval_script="$SEL_EVAL_SCRIPT" ;;
        *)
            echo "[ERROR] Unknown eval_key='$eval_key' for job $train_job_id"
            exit 1
            ;;
    esac

    run_dir="$TRAIN_SAVE_ROOT/${train_job_name}_${train_job_id}"
    output_path="$EVAL_OUTPUT_ROOT/${train_job_name}_${train_job_id}"
    eval_job_name="eval_${train_job_name}"
    run_name="${train_job_name}_${train_job_id}_vsibench"
    note="VSI-Bench eval for ${train_job_name} (dep on ${train_job_id})"

    echo "[SUBMIT] train_job=$train_job_id name=$train_job_name eval_script=$(basename "$eval_script")"
    echo "         run_dir=$run_dir"

    submit_output=$(
        PRETRAINED_LOCAL="$run_dir" \
        MODEL_BASE_LOCAL="$MODEL_BASE_LOCAL" \
        SIGLIP_LOCAL="$SIGLIP_LOCAL" \
        MODEL_NAME="$MODEL_NAME" \
        CONV_TEMPLATE="$CONV_TEMPLATE" \
        MAX_FRAMES_NUM="$MAX_FRAMES_NUM" \
        RUN_NAME="$run_name" \
        OUTPUT_PATH="$output_path" \
        NOTE="$note" \
        LIMIT="$LIMIT" \
        sbatch \
            --dependency "afterok:${train_job_id}" \
            --job-name "$eval_job_name" \
            --export=ALL,PRETRAINED_LOCAL,MODEL_BASE_LOCAL,SIGLIP_LOCAL,MODEL_NAME,CONV_TEMPLATE,MAX_FRAMES_NUM,RUN_NAME,OUTPUT_PATH,NOTE,LIMIT \
            "$eval_script"
    )
    echo "         $submit_output"
    echo ""
done

echo "[DONE] Submitted ${#JOBS[@]} evaluation jobs."
