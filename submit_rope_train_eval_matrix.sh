#!/bin/bash
set -euo pipefail

TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_rope_cut3r.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_rope_cut3r.sh}"
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"

submit_pair() {
  local train_job_name="$1"
  local fusion_block="$2"
  local rope_mode="$3"
  local group_split="$4"

  local train_job_id
  local model_path
  local eval_job_name

  echo "[TRAIN SUBMIT] name=$train_job_name fusion=$fusion_block mode=$rope_mode split=$group_split"
  train_job_id="$(
    export MODEL_FUSION_BLOCK="$fusion_block"
    export MODEL_GEOMETRY_ROPE_MODE="$rope_mode"
    export MODEL_GEOMETRY_ROPE_GROUP_SPLIT="$group_split"
    export MODEL_GEOMETRY_ROPE_MAX_DEPTH="10.0"
    export MODEL_GEOMETRY_ROPE_LOG_STATS="False"
    export TRAIN_DATA_PERCENTAGE="100"
    sbatch --parsable --job-name "$train_job_name" --export=ALL "$TRAIN_SCRIPT"
  )"

  model_path="$TRAIN_SAVE_ROOT/${train_job_name}_${train_job_id}"
  eval_job_name="eval_${train_job_name}_${train_job_id}"

  echo "[EVAL SUBMIT]  name=$eval_job_name dependency=afterok:$train_job_id"
  echo "               model_path=$model_path"
  (
    export PRETRAINED_LOCAL="$model_path"
    export RUN_NAME="$eval_job_name"
    export RUNTIME_ROOT="/leonardo_scratch/fast/EUHPC_D32_006/eval/runtime/$eval_job_name"
    export MODEL_FUSION_BLOCK="$fusion_block"
    export MODEL_GEOMETRY_ROPE_MODE="$rope_mode"
    export MODEL_GEOMETRY_ROPE_GROUP_SPLIT="$group_split"
    export MODEL_GEOMETRY_ROPE_MAX_DEPTH="10.0"
    export MODEL_GEOMETRY_ROPE_LOG_STATS="False"
    export SPATIAL_FEATURES_ROOT="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r"
    export SPATIAL_FEATURES_SUBDIR="spatial_features_points"
    sbatch --parsable --job-name "$eval_job_name" --dependency="afterok:$train_job_id" --export=ALL "$EVAL_SCRIPT"
  )

  echo ""
}

submit_pair "rope_depth_100p" "svf_depth_rope" "depth" "1"
submit_pair "rope_xyz_100p" "svf_xyz_rope" "xyz" "2,1,2"
submit_pair "rope_spherical_100p" "svf_spherical_rope" "spherical" "2,1,2"
