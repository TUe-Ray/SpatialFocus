#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/train_cut3r_layer_ablation.sh}"
CUT3R_LAYER="${CUT3R_LAYER:--2}"
EXTRACT_JOB_IDS="${EXTRACT_JOB_IDS:-}"
FEATURE_ROOT="${FEATURE_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
DRY_RUN="${DRY_RUN:-False}"

case "$CUT3R_LAYER" in
    -2|m2|dec_m2|spatial_features_dec_m2)
        CUT3R_LAYER="-2"
        FEATURE_SUBDIR="spatial_features_dec_m2"
        ;;
    -4|m4|dec_m4|spatial_features_dec_m4)
        CUT3R_LAYER="-4"
        FEATURE_SUBDIR="spatial_features_dec_m4"
        ;;
    -1|m1|dec_m1|spatial_features)
        CUT3R_LAYER="-1"
        FEATURE_SUBDIR="spatial_features"
        ;;
    *)
        echo "[ERROR] Unsupported CUT3R_LAYER='$CUT3R_LAYER'. Use -1, -2, or -4."
        exit 1
        ;;
esac

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[ERROR] Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

feature_dir_ready() {
    local dataset="$1"
    local dir="$FEATURE_ROOT/$dataset/$FEATURE_SUBDIR"
    [[ -d "$dir" ]] || return 1
    [[ -n "$(find "$dir" -maxdepth 1 -type f -name '*.pt' -print -quit)" ]] || return 1
}

dependency_ids_for_job_name() {
    local job_name="$1"
    squeue -h -u "${USER:-$LOGNAME}" -n "$job_name" -o "%i" | tr '\n' ':' | sed 's/:$//'
}

append_dependency_ids() {
    local ids="$1"
    if [[ -n "$ids" ]]; then
        if [[ -n "$EXTRACT_JOB_IDS" ]]; then
            EXTRACT_JOB_IDS="$EXTRACT_JOB_IDS:$ids"
        else
            EXTRACT_JOB_IDS="$ids"
        fi
    fi
}

if [[ "$CUT3R_LAYER" != "-1" && -z "$EXTRACT_JOB_IDS" ]]; then
    declare -A EXTRACT_JOB_NAMES=(
        [scannet]="vlm3r_scannet_cut3r_dec_m2m4"
        [scannetpp]="vlm3r_scannetpp_cut3r_dec_m2m4"
        [arkitscenes]="vlm3r_arkitscenes_cut3r_dec_m2m4"
    )
    for dataset in scannet scannetpp arkitscenes; do
        ids="$(dependency_ids_for_job_name "${EXTRACT_JOB_NAMES[$dataset]}")"
        if [[ -n "$ids" ]]; then
            append_dependency_ids "$ids"
            echo "[INFO] $dataset extraction still active; adding dependency on job(s): $ids"
        elif feature_dir_ready "$dataset"; then
            echo "[INFO] $dataset features already populated for $FEATURE_SUBDIR."
        else
            echo "[ERROR] $dataset has no active extraction job and no populated feature dir for $FEATURE_SUBDIR."
            echo "[ERROR] Expected directory: $FEATURE_ROOT/$dataset/$FEATURE_SUBDIR"
            echo "[ERROR] Submit the extraction job first, or pass EXTRACT_JOB_IDS='job1:job2:job3'."
            exit 1
        fi
    done
fi

SBATCH_ARGS=()
if [[ "$CUT3R_LAYER" == "-1" ]]; then
    echo "[INFO] CUT3R_LAYER=-1 uses existing baseline spatial_features; no extraction dependency is needed."
elif [[ -n "$EXTRACT_JOB_IDS" ]]; then
    SBATCH_ARGS+=("--dependency=afterok:$EXTRACT_JOB_IDS")
    echo "[INFO] Training will wait for extraction jobs: $EXTRACT_JOB_IDS"
else
    echo "[INFO] No active extraction jobs found, and all feature directories for $FEATURE_SUBDIR are populated."
    echo "[INFO] Submitting training without a Slurm dependency because extraction appears complete."
fi

echo "[INFO] Submitting $TRAIN_SCRIPT with CUT3R_LAYER=$CUT3R_LAYER"
if [[ "$DRY_RUN" == "True" || "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] sbatch'
    for arg in "${SBATCH_ARGS[@]}"; do
        printf ' %q' "$arg"
    done
    printf ' --export=%q %q\n' "ALL,CUT3R_LAYER=$CUT3R_LAYER" "$TRAIN_SCRIPT"
    exit 0
fi

sbatch "${SBATCH_ARGS[@]}" --export=ALL,CUT3R_LAYER="$CUT3R_LAYER" "$TRAIN_SCRIPT"
