#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_SCRIPT="${BASE_SCRIPT:-$PROJECT_ROOT/train_eomt_cut3r_obj.sh}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "[ERROR] Base script not found: $BASE_SCRIPT"
    exit 1
fi

JOB_PREFIX="${JOB_PREFIX:-cut3r_eomt_obj}"
TRAIN_DATA_PERCENTAGE="${TRAIN_DATA_PERCENTAGE:-50}"
MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-cross_attention}"

MODES=(
    "baseline"
    "eomt_obj_only"
    "eomt_obj_text_phrase"
    "eomt_obj_learnable"
    "eomt_obj_only_keep_stuff"
    "eomt_obj_only_word_filter"
)

short_mode_name() {
    case "$1" in
        baseline) echo "base" ;;
        eomt_obj_only) echo "obj" ;;
        eomt_obj_text_phrase) echo "objtext" ;;
        eomt_obj_learnable) echo "objlearn" ;;
        eomt_obj_only_keep_stuff) echo "objstuff" ;;
        eomt_obj_only_word_filter) echo "objword" ;;
        *) echo "unknown" ;;
    esac
}

echo "[SUBMIT] base_script=$BASE_SCRIPT"
echo "[SUBMIT] job_prefix=$JOB_PREFIX"
echo "[SUBMIT] train_data_percentage=$TRAIN_DATA_PERCENTAGE"
echo "[SUBMIT] model_fusion_block=$MODEL_FUSION_BLOCK"
echo "[SUBMIT] modes=${MODES[*]}"

for mode in "${MODES[@]}"; do
    short_name="$(short_mode_name "$mode")"
    job_name="${JOB_PREFIX}_${short_name}"
    note="CUT3R + EoMT object-token mode=${mode} fusion=${MODEL_FUSION_BLOCK} data=${TRAIN_DATA_PERCENTAGE}% epoch=1"

    echo "[SUBMIT] mode=$mode job_name=$job_name"
    submit_output=$(
        EOMT_EXPERIMENT_MODE="$mode" \
        NOTE="$note" \
        TRAIN_DATA_PERCENTAGE="$TRAIN_DATA_PERCENTAGE" \
        MODEL_FUSION_BLOCK="$MODEL_FUSION_BLOCK" \
        sbatch \
            --job-name="$job_name" \
            --export=ALL,EOMT_EXPERIMENT_MODE,NOTE,TRAIN_DATA_PERCENTAGE,MODEL_FUSION_BLOCK \
            "$BASE_SCRIPT"
    )
    echo "[SUBMIT] $submit_output"
done

echo "[DONE] Submitted all ${#MODES[@]} EoMT object-token modes."
