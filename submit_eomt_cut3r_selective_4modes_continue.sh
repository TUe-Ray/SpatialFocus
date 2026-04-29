#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_SCRIPT="${BASE_SCRIPT:-$PROJECT_ROOT/train_eomt_cut3r_selective_fusion_Continue.sh}"
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
JOB_PREFIX="${JOB_PREFIX:-cut3r_eomt_sel3d_resume}"
TRAIN_DATA_PERCENTAGE="${TRAIN_DATA_PERCENTAGE:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "[ERROR] Base script not found: $BASE_SCRIPT"
    exit 1
fi

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

find_latest_resumable_checkpoint() {
    local run_dir="$1"
    local checkpoint_dir
    local -a checkpoint_dirs=()

    mapfile -t checkpoint_dirs < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V -r)

    for checkpoint_dir in "${checkpoint_dirs[@]}"; do
        # A complete resumable checkpoint needs trainer state, RNG state, and DeepSpeed shard metadata.
        if [[ -f "$checkpoint_dir/trainer_state.json" ]] \
            && [[ -f "$checkpoint_dir/latest" ]] \
            && compgen -G "$checkpoint_dir/global_step*" > /dev/null; then
            echo "$checkpoint_dir"
            return 0
        fi
    done

    return 1
}

echo "[SUBMIT] base_script=$BASE_SCRIPT"
echo "[SUBMIT] train_save_root=$TRAIN_SAVE_ROOT"
echo "[SUBMIT] job_prefix=$JOB_PREFIX"
echo "[SUBMIT] train_data_percentage=$TRAIN_DATA_PERCENTAGE"
echo "[SUBMIT] save_steps=$SAVE_STEPS"
echo "[SUBMIT] modes=${MODES[*]}"

for mode in "${MODES[@]}"; do
    run_name="${RUN_DIRS[$mode]}"
    run_dir="$TRAIN_SAVE_ROOT/$run_name"
    if [[ ! -d "$run_dir" ]]; then
        echo "[ERROR] Run directory not found for mode=$mode: $run_dir"
        exit 1
    fi

    if ! checkpoint_dir="$(find_latest_resumable_checkpoint "$run_dir")"; then
        echo "[ERROR] No resumable checkpoint found for mode=$mode in $run_dir"
        exit 1
    fi

    short_name="$(short_mode_name "$mode")"
    job_name="${JOB_PREFIX}_${short_name}"
    note="Resume CUT3R + EoMT selective 3D mode=${mode} from $(basename "$checkpoint_dir") data=${TRAIN_DATA_PERCENTAGE}% save_steps=${SAVE_STEPS}"

    echo "[SUBMIT] mode=$mode job_name=$job_name checkpoint=$checkpoint_dir"
    submit_output=$(
        EOMT_EXPERIMENT_MODE="$mode" \
        NOTE="$note" \
        TRAIN_DATA_PERCENTAGE="$TRAIN_DATA_PERCENTAGE" \
        RESUME_MODE="continue" \
        RESUME_CHECKPOINT_PATH="$checkpoint_dir" \
        MID_RUN_NAME_OVERRIDE="$run_name" \
        OUTPUT_DIR_OVERRIDE="$run_dir" \
        RUN_NAME_OVERRIDE="$run_name" \
        SAVE_STEPS="$SAVE_STEPS" \
        sbatch \
            --job-name="$job_name" \
            --export=ALL,EOMT_EXPERIMENT_MODE,NOTE,TRAIN_DATA_PERCENTAGE,RESUME_MODE,RESUME_CHECKPOINT_PATH,MID_RUN_NAME_OVERRIDE,OUTPUT_DIR_OVERRIDE,RUN_NAME_OVERRIDE,SAVE_STEPS \
            "$BASE_SCRIPT"
    )
    echo "[SUBMIT] $submit_output"
done

echo "[DONE] Submitted all 4 EoMT selective resume jobs."