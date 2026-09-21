#!/usr/bin/env bash
#SBATCH --job-name=vggt_l23_full
#SBATCH --account=tesei2748
#SBATCH --partition=rome
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_artifacts/logs/%x_%j.out
#SBATCH --error=/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_artifacts/logs/%x_%j.err

set -Eeuo pipefail

readonly SCRIPT_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus/snellius/vggt_l23"
source "$SCRIPT_DIR/runtime.sh"

vggt_l23_preflight
run_vggt_l23_conversion full --publish
echo "VGGT L23 full conversion, inventory, validation, and publication passed."
