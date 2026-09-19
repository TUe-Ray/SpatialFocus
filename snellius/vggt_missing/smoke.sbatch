#!/usr/bin/env bash
#SBATCH --job-name=vggt_missing_smoke
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_logs/%x_%j.out
#SBATCH --error=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging/_logs/%x_%j.err

set -Eeuo pipefail

readonly SCRIPT_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus/snellius/vggt_missing"
source "$SCRIPT_DIR/runtime.sh"

vggt_runtime_preflight
run_vggt_extract --input-file "$EVAL_SCANNET_VIEW/scene0025_01.mp4" "$SCANNET_OUTPUT"
run_vggt_extract --input-file "$TRAIN_REPAIR_VIEW/dd685be466.mp4" "$SCANNETPP_OUTPUT"
run_vggt_extract --input-file "$EVAL_ARKITSCENES_VIEW/41069025.mp4" "$ARKITSCENES_OUTPUT"

python "$VALIDATOR" \
  --spec smoke_scannet "$SCRIPT_DIR/smoke_scannet_ids.txt" "$RAW_VSI_ROOT/scannet" "$SCANNET_OUTPUT" \
  --spec smoke_scannetpp "$SCRIPT_DIR/smoke_scannetpp_ids.txt" "$RAW_TRAIN_ROOT/scannetpp/videos" "$SCANNETPP_OUTPUT" \
  --spec smoke_arkitscenes "$SCRIPT_DIR/smoke_arkitscenes_ids.txt" "$RAW_VSI_ROOT/arkitscenes" "$ARKITSCENES_OUTPUT" \
  --reference-sidecar "$FINAL_ROOT/scannet/scene0261_01.pt" \
  --report "$REPORT_ROOT/smoke_${SLURM_JOB_ID}.json"

echo "VGGT three-dataset compatibility smoke passed."
