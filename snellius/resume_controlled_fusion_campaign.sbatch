#!/usr/bin/env bash
# Resume one official B/C/D/E/H task from an exact protected milestone.
# Required exports: CONTROLLED_FUSION_ID, CONTROLLED_FUSION_RUN_TOKEN_OVERRIDE,
# and CONTROLLED_FUSION_RESUME_CHECKPOINT_PATH.
#SBATCH --job-name=controlled_fusion_resume_4n16g
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/train/%x_%j.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/train/%x_%j.err

set -Eeuo pipefail

: "${CONTROLLED_FUSION_ID:?Set the controlled architecture ID.}"
: "${CONTROLLED_FUSION_RUN_TOKEN_OVERRIDE:?Set the original array token, for example 26484327_1.}"
: "${CONTROLLED_FUSION_RESUME_CHECKPOINT_PATH:?Set the exact checkpoint directory.}"
export CONTROLLED_FUSION_RESUME_MODE=continue

exec bash /gpfs/home4/geusdd/shuang/SpatialFocus/snellius/train_controlled_fusion_campaign.sbatch
