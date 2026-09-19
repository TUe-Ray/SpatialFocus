#!/usr/bin/env bash
#SBATCH --job-name=vggt_final_audit
#SBATCH --account=tesei2748
#SBATCH --partition=rome
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_logs/%x_%j.out
#SBATCH --error=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_logs/%x_%j.err

set -Eeuo pipefail

readonly REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
readonly VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
export PYTHONNOUSERSITE=1
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"
export MPLCONFIGDIR="/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_runtime_cache/matplotlib"

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname -f)"
echo "slurm_job_id=$SLURM_JOB_ID"
python "$REPO_DIR/scripts/extraction/audit_vggt_final.py"
