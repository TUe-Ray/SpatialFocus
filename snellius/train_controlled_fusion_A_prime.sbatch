#!/usr/bin/env bash
# Official single-variant A-prime production run; architecture and recipe live
# in the shared controlled-fusion launchers.
#SBATCH --job-name=controlled_A_prime_4n16g
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --mem=0
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/train/%x_%j.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/train/%x_%j.err

set -Eeuo pipefail

REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
export CONTROLLED_FUSION_ID=A_prime
exec bash "$REPO_DIR/snellius/train_controlled_fusion_campaign.sbatch"
