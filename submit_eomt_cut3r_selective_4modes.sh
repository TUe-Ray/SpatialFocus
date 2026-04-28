#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_SCRIPT="${BASE_SCRIPT:-$PROJECT_ROOT/train_eomt_cut3r_selective_fusion.sh}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "[ERROR] Base script not found: $BASE_SCRIPT"
    exit 1
fi

JOB_PREFIX="${JOB_PREFIX:-cut3r_eomt_sel3d}"
TRAIN_DATA_PERCENTAGE="${TRAIN_DATA_PERCENTAGE:-50}"

MODES=(
    "baseline"
    "selective_soft"
    "selective_soft_with_floor"
    "selective_soft_with_floor_zero_fallback"
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

echo "[SUBMIT] base_script=$BASE_SCRIPT"
echo "[SUBMIT] job_prefix=$JOB_PREFIX"
echo "[SUBMIT] train_data_percentage=$TRAIN_DATA_PERCENTAGE"
echo "[SUBMIT] modes=${MODES[*]}"

for mode in "${MODES[@]}"; do
    short_name="$(short_mode_name "$mode")"
    job_name="${JOB_PREFIX}_${short_name}"
    note="CUT3R + EoMT selective 3D mode=${mode} data=${TRAIN_DATA_PERCENTAGE}% epoch=1"

    echo "[SUBMIT] mode=$mode job_name=$job_name"
    submit_output=$(
        EOMT_EXPERIMENT_MODE="$mode" \
        NOTE="$note" \
        TRAIN_DATA_PERCENTAGE="$TRAIN_DATA_PERCENTAGE" \
        sbatch \
            --job-name="$job_name" \
            --export=ALL,EOMT_EXPERIMENT_MODE,NOTE,TRAIN_DATA_PERCENTAGE \
            "$BASE_SCRIPT"
    )
    echo "[SUBMIT] $submit_output"
done

echo "[DONE] Submitted all 4 EoMT selective modes."
