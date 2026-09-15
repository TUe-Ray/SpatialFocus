#!/usr/bin/env bash
# Official VSI-Bench evaluator for a single controlled A-prime training job.
# TRAIN_JOB_ID identifies the producing non-array Slurm job.
#SBATCH --job-name=eval_controlled_A_prime
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/eval/%x_%j.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/eval/%x_%j.err

set -Eeuo pipefail

: "${TRAIN_JOB_ID:?Set TRAIN_JOB_ID to the producing A_prime Slurm job.}"
REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
export CONTROLLED_FUSION_ID=A_prime
exec bash "$REPO_DIR/snellius/eval_controlled_fusion_campaign.sbatch"
