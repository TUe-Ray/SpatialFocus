#!/usr/bin/env bash
#SBATCH --job-name=vggt_merge_when_rsync_stable
#SBATCH --account=tesei2748
#SBATCH --partition=rome
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_logs/%x_%j.out
#SBATCH --error=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_logs/%x_%j.err

set -Eeuo pipefail

readonly REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
readonly VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
readonly MERGER="$REPO_DIR/scripts/extraction/merge_vggt_missing_staging.py"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname -f)"
echo "slurm_job_id=$SLURM_JOB_ID"
echo "Waiting for the exact 2,281-file Leonardo rsync coverage."

while ! python "$MERGER" --record-stability; do
  echo "$(date --iso-8601=seconds) rsync incomplete; retrying in 60 seconds"
  sleep 60
done

echo "Exact pre-existing coverage recorded; waiting 600 seconds for stability."
sleep 600
python "$MERGER" --merge
echo "Validated staging merge completed."
