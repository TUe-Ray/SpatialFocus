#!/usr/bin/env bash
# Submit independent controlled-fusion eval tasks after their matching train
# task exits.  The eval launcher verifies the final root checkpoint and all
# required milestone artifacts before it consumes any GPU evaluation work.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ARRAY_JOB_ID="${1:?Usage: $0 TRAIN_ARRAY_JOB_ID [TASK_INDEX ...]}"
shift
[[ "$TRAIN_ARRAY_JOB_ID" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAIN_ARRAY_JOB_ID must be numeric, got: $TRAIN_ARRAY_JOB_ID" >&2
    exit 2
}

if (( $# == 0 )); then
    set -- 0 1 2 3 4
fi

ARCH_IDS=(B C D E H)
for array_index in "$@"; do
    [[ "$array_index" =~ ^[0-4]$ ]] || {
        echo "Unsupported controlled-fusion task index: $array_index" >&2
        exit 2
    }
    arch_id="${ARCH_IDS[$array_index]}"
    job_id="$(sbatch --parsable \
        --job-name="eval_controlled_${arch_id}_after_train_${TRAIN_ARRAY_JOB_ID}" \
        --array="$array_index" \
        --dependency="afterany:${TRAIN_ARRAY_JOB_ID}_${array_index}" \
        --export="ALL,TRAIN_ARRAY_JOB_ID=${TRAIN_ARRAY_JOB_ID}" \
        "$REPO_DIR/snellius/eval_controlled_fusion_campaign.sbatch")"
    printf 'submitted arch=%s train_task=%s_%s eval_job=%s\n' \
        "$arch_id" "$TRAIN_ARRAY_JOB_ID" "$array_index" "$job_id"
done
