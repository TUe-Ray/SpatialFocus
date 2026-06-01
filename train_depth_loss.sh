#!/bin/bash
#SBATCH --job-name=cut3r_depth_loss
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NOTE="${NOTE:-Train VLM3R on VSI-Bench with CUT3R token sidecars, CUT3R point-map sidecars, cross-attention fusion, and depth-only auxiliary supervision.}"
export MODEL_USE_BEV_SUPERVISION="${MODEL_USE_BEV_SUPERVISION:-False}"
export MODEL_USE_DEPTH_SUPERVISION="${MODEL_USE_DEPTH_SUPERVISION:-True}"
export MODEL_LAMBDA_DEPTH="${MODEL_LAMBDA_DEPTH:-0.05}"
export MODEL_DEPTH_POINT_MAP_KEY="${MODEL_DEPTH_POINT_MAP_KEY:-point_maps_cam}"
export MODEL_DEPTH_DETACH_HIDDEN="${MODEL_DEPTH_DETACH_HIDDEN:-False}"
export MODEL_DEPTH_SHUFFLE_TARGET="${MODEL_DEPTH_SHUFFLE_TARGET:-False}"
export MODEL_DEPTH_SHUFFLE_MODE="${MODEL_DEPTH_SHUFFLE_MODE:-frame_shuffle}"
export MODEL_DEPTH_CONF_THRESHOLD="${MODEL_DEPTH_CONF_THRESHOLD:-0.0}"
export MODEL_DEPTH_MAX_GT="${MODEL_DEPTH_MAX_GT:-20.0}"
export MODEL_DEPTH_ALLOW_GENERIC_CAMERA_ASSUMED="${MODEL_DEPTH_ALLOW_GENERIC_CAMERA_ASSUMED:-False}"
export MODEL_DEPTH_ALLOW_TENSOR_CAMERA_ASSUMED="${MODEL_DEPTH_ALLOW_TENSOR_CAMERA_ASSUMED:-False}"
export MODEL_DEPTH_HEAD_SOURCE="${MODEL_DEPTH_HEAD_SOURCE:-llm_output}"
export MODEL_DEPTH_VISUALIZE_DEBUG="${MODEL_DEPTH_VISUALIZE_DEBUG:-False}"

export GEOMETRY_SPATIAL_TOWER_TYPE="${GEOMETRY_SPATIAL_TOWER_TYPE:-cut3r}"
export REQUIRE_GEOMETRY_SPATIAL_FEATURES="${REQUIRE_GEOMETRY_SPATIAL_FEATURES:-True}"
export GEOMETRY_SPATIAL_FEATURES_SUBDIR="${GEOMETRY_SPATIAL_FEATURES_SUBDIR:-spatial_features_points}"

exec "${SCRIPT_DIR}/train_BEV_loss.sh" "$@"
