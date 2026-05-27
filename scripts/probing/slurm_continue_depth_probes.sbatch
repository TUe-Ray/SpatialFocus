#!/bin/bash
#SBATCH --job-name=vlm3r_depth_continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --output=logs/depth_probe_continue_%x_%j.out
#SBATCH --error=logs/depth_probe_continue_%x_%j.err

set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK:-/leonardo_work/EUHPC_D32_006}/probing_experiment}"
export MODEL_LABELS="${MODEL_LABELS:-zero_spatial,vlm3r_baseline,geo_rope_fusion}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_NAME="${WANDB_NAME:-depth-probes-continue-${SLURM_JOB_ID:-manual}}"

bash scripts/probing/slurm_train_depth_probes.sbatch
