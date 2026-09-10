#!/usr/bin/env bash
# Train the existing A/F/G references through the protected 50%-of-epoch
# milestone. Array indices: 0=A (VLM3R), 1=F (SpatialStack Add), 2=G
# (SpatialStack Cross-Attn).
#SBATCH --job-name=reference_fusion_AFG_half_4n16g
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --mem=0
#SBATCH --array=0-2%3
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/train/%x_%A_%a.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/train/%x_%A_%a.err

set -Eeuo pipefail

REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
ARCH_IDS=(A F G)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array task.}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#ARCH_IDS[@]} )); then
    echo "Unsupported reference array index: $ARRAY_INDEX" >&2
    exit 2
fi
export REFERENCE_FUSION_ID="${ARCH_IDS[$ARRAY_INDEX]}"

exec bash "$REPO_DIR/snellius/train_controlled_fusion_campaign.sbatch"
