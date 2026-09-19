#!/usr/bin/env bash
#SBATCH --job-name=vggt_missing_scannet
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_logs/%x_%j.out
#SBATCH --error=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_logs/%x_%j.err

set -Eeuo pipefail

readonly SCRIPT_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus/snellius/vggt_missing"
source "$SCRIPT_DIR/runtime.sh"

vggt_runtime_preflight
run_vggt_extract --input-dir "$EVAL_SCANNET_VIEW" "$SCANNET_OUTPUT"
python "$VALIDATOR" \
  --spec eval_scannet "$EVAL_SCANNET_IDS" "$RAW_VSI_ROOT/scannet" "$SCANNET_OUTPUT" \
  --report "$REPORT_ROOT/eval_scannet_${SLURM_JOB_ID}.json"
