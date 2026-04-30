#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_SCRIPT="${BASE_SCRIPT:-$PROJECT_ROOT/train_eomt_cut3r_selective_fusion.sh}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "[ERROR] Base script not found: $BASE_SCRIPT"
    exit 1
fi

JOB_PREFIX="${JOB_PREFIX:-cut3r_eomt_sel3dr2}"
TRAIN_DATA_PERCENTAGE="${TRAIN_DATA_PERCENTAGE:-100}"
EOMT_EXPERIMENT_CONFIG_PATH="${EOMT_EXPERIMENT_CONFIG_PATH:-/leonardo/home/userexternal/shuang00/VLM-3R/configs/eomt/eomt_selective_3d_round2.json}"

MODES=(
    "soft_word_match_all3d"
    "soft_word_match_zero3d"
    "soft_no_word_match_all3d"
    "soft_no_word_match_zero3d"
)

short_mode_name() {
    case "$1" in
        soft_word_match_all3d) echo "wmall" ;;
        soft_word_match_zero3d) echo "wmzero" ;;
        soft_no_word_match_all3d) echo "nowmall" ;;
        soft_no_word_match_zero3d) echo "nowmzero" ;;
        *) echo "unknown" ;;
    esac
}

echo "[SUBMIT] base_script=$BASE_SCRIPT"
echo "[SUBMIT] job_prefix=$JOB_PREFIX"
echo "[SUBMIT] train_data_percentage=$TRAIN_DATA_PERCENTAGE"
echo "[SUBMIT] eomt_experiment_config_path=$EOMT_EXPERIMENT_CONFIG_PATH"
echo "[SUBMIT] modes=${MODES[*]}"

for mode in "${MODES[@]}"; do
    short_name="$(short_mode_name "$mode")"
    job_name="${JOB_PREFIX}_${short_name}"
    note="CUT3R + EoMT selective 3D mode=${mode} data=${TRAIN_DATA_PERCENTAGE}% epoch=1"

    echo "[SUBMIT] mode=$mode job_name=$job_name"
    submit_output=$(
        EOMT_EXPERIMENT_MODE="$mode" \
        EOMT_EXPERIMENT_CONFIG_PATH="$EOMT_EXPERIMENT_CONFIG_PATH" \
        NOTE="$note" \
        TRAIN_DATA_PERCENTAGE="$TRAIN_DATA_PERCENTAGE" \
        sbatch \
            --job-name="$job_name" \
            --export=ALL,EOMT_EXPERIMENT_MODE,EOMT_EXPERIMENT_CONFIG_PATH,NOTE,TRAIN_DATA_PERCENTAGE \
            "$BASE_SCRIPT"
    )
    echo "[SUBMIT] $submit_output"
done

echo "[DONE] Submitted all 4 EoMT selective modes."
