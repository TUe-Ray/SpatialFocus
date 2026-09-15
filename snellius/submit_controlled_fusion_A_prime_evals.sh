#!/usr/bin/env bash
# Submit the canonical final and milestone VSI evaluations after A-prime SFT.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_JOB_ID="${1:?Usage: $0 TRAIN_JOB_ID}"
[[ "$TRAIN_JOB_ID" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAIN_JOB_ID must be numeric, got: $TRAIN_JOB_ID" >&2
    exit 2
}

for stage in final p01 p05 p25 p50; do
    job_id="$(sbatch --parsable \
        --job-name="eval_controlled_A_prime_${stage}_after_${TRAIN_JOB_ID}" \
        --dependency="afterany:${TRAIN_JOB_ID}" \
        --export="ALL,TRAIN_JOB_ID=${TRAIN_JOB_ID},FUSION_EVAL_STAGE=${stage}" \
        "$REPO_DIR/snellius/eval_controlled_fusion_A_prime.sbatch")"
    printf 'submitted arch=A_prime stage=%s train_job=%s eval_job=%s\n' \
        "$stage" "$TRAIN_JOB_ID" "$job_id"
done
