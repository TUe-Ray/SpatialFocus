#!/usr/bin/env bash
# Submit A/F/G half-epoch training and independent milestone evaluations.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
train_job_id="$(sbatch --parsable "$REPO_DIR/snellius/train_reference_fusion_half_campaign.sbatch")"
printf 'submitted reference A/F/G training array=%s\n' "$train_job_id"

ARCH_IDS=(A F G)
STAGES=(p01 p05 p25 p50)
for array_index in 0 1 2; do
    arch_id="${ARCH_IDS[$array_index]}"
    for stage in "${STAGES[@]}"; do
        eval_job_id="$(sbatch --parsable \
            --job-name="eval_reference_${arch_id}_${stage}_after_${train_job_id}" \
            --array="$array_index" \
            --dependency="afterany:${train_job_id}_${array_index}" \
            --export="ALL,FUSION_CAMPAIGN_KIND=reference_half,TRAIN_ARRAY_JOB_ID=${train_job_id},FUSION_EVAL_STAGE=${stage}" \
            "$REPO_DIR/snellius/eval_controlled_fusion_campaign.sbatch")"
        printf 'submitted arch=%s stage=%s train_task=%s_%s eval_job=%s\n' \
            "$arch_id" "$stage" "$train_job_id" "$array_index" "$eval_job_id"
    done
done
