#!/usr/bin/env bash
# Submit A/F/G half-epoch training and independent milestone evaluations.
# Legacy A exceeds the 40-GiB A100 memory envelope, so it runs on a 80-GiB
# H100 node. F/G retain the established A100 recipe.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
a_train_job_id="$(sbatch --parsable \
    --partition=gpu_h100 \
    --cpus-per-task=64 \
    --array=0 \
    --job-name=reference_fusion_A_half_4n16g_h100 \
    "$REPO_DIR/snellius/train_reference_fusion_half_campaign.sbatch")"
fg_train_job_id="$(sbatch --parsable \
    --array=1-2%2 \
    --job-name=reference_fusion_FG_half_4n16g_a100 \
    "$REPO_DIR/snellius/train_reference_fusion_half_campaign.sbatch")"
printf 'submitted reference A H100 training array=%s\n' "$a_train_job_id"
printf 'submitted reference F/G A100 training array=%s\n' "$fg_train_job_id"

ARCH_IDS=(A F G)
STAGES=(p01 p05 p25 p50)
for array_index in 0 1 2; do
    arch_id="${ARCH_IDS[$array_index]}"
    if (( array_index == 0 )); then
        train_job_id="$a_train_job_id"
    else
        train_job_id="$fg_train_job_id"
    fi
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
