#!/bin/bash
# Re-submit eval for rope_depth and rope_xyz whose previous runs were corrupted
# by the shared RUNTIME_ROOT race condition (job IDs 40951364 and 40951368).
# Both jobs loaded rope_spherical_100p_40790070 weights by mistake.
#
# Usage: bash submit_reeval_depth_xyz.sh

set -euo pipefail

EVAL_SCRIPT="${EVAL_SCRIPT:-eval_geo_rope_fusion_cut3r.sh}"
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"

submit_eval() {
  local pretrained_dir="$1"
  local run_name="$2"
  local fusion_block="$3"
  local rope_mode="$4"
  local group_split="$5"

  echo "[EVAL SUBMIT] run_name=$run_name"
  echo "              pretrained=$pretrained_dir"

  local job_id
  job_id="$(
    export PRETRAINED_LOCAL="$pretrained_dir"
    export RUN_NAME="$run_name"
    export MODEL_FUSION_BLOCK="$fusion_block"
    export MODEL_GEO_ROPE_FUSION_MODE="$rope_mode"
    export MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="$group_split"
    export MODEL_GEO_ROPE_FUSION_MAX_DEPTH="10.0"
    export MODEL_GEO_ROPE_FUSION_LOG_STATS="False"
    sbatch --parsable --job-name "$run_name" --export=ALL "$EVAL_SCRIPT"
  )"
  echo "  -> submitted job $job_id"
  echo ""
}

# rope_depth — was job 40951364, loaded spherical weights by mistake
submit_eval \
  "$TRAIN_SAVE_ROOT/rope_depth_100p_40790065" \
  "eval_rope_depth_100p_40790065" \
  "svf_depth_rope" \
  "depth" \
  "1"

# rope_xyz — was job 40951368, loaded spherical weights by mistake
submit_eval \
  "$TRAIN_SAVE_ROOT/rope_xyz_100p_40790067" \
  "eval_rope_xyz_100p_40790067" \
  "svf_xyz_rope" \
  "xyz" \
  "2,1,2"
