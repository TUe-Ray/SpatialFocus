#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_DIR/eval_RoPE_cut3r.sh}"

OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/RoPE_cut3r_sidecars}"
LIMIT="${LIMIT:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

EVAL_TIME="${EVAL_TIME:-04:00:00}"
EVAL_PARTITION="${EVAL_PARTITION:-boost_usr_prod}"
EVAL_QOS="${EVAL_QOS:-normal}"
DRY_RUN="${DRY_RUN:-False}"

MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"

RUN_DEPTH="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/RoPE_Depth_cut3r_100p_41519216"
RUN_SPHERICAL="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/RoPE_Spherical_cut3r_100p_41520134"
RUN_XYZ="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/RoPE_XYZ_cut3r_100p_41519421"

mkdir -p "$REPO_DIR/logs/eval" "$OUTPUT_PATH"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] Missing eval script: $EVAL_SCRIPT"
  exit 1
fi
if [[ ! -f "$CUT3R_WEIGHTS" ]]; then
  echo "[ERROR] Missing CUT3R weights: $CUT3R_WEIGHTS"
  exit 1
fi
for path in "$MODEL_BASE_LOCAL" "$SIGLIP_LOCAL" "$RUN_DEPTH" "$RUN_SPHERICAL" "$RUN_XYZ"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
done

submit_one() {
  local tag="$1"
  local mode="$2"
  local ckpt="$3"

  if [[ ! -f "$ckpt/config.json" ]]; then
    echo "[ERROR] Missing checkpoint config: $ckpt/config.json"
    exit 1
  fi

  local job_name="Eval_${mode}_cut3r"
  echo "==== Submit $tag ===="
  echo "checkpoint=$ckpt"
  echo "geometry_mode=$mode"
  echo "job_name=$job_name"
  echo "output_path=$OUTPUT_PATH"

  (
    export PRETRAINED_LOCAL="$ckpt"
    export MODEL_BASE_LOCAL="$MODEL_BASE_LOCAL"
    export SIGLIP_LOCAL="$SIGLIP_LOCAL"
    export CUT3R_WEIGHTS="$CUT3R_WEIGHTS"
    export RUN_NAME="$tag"
    export OUTPUT_PATH="$OUTPUT_PATH"
    export LIMIT="$LIMIT"
    export NUM_PROCESSES="$NUM_PROCESSES"
    export BATCH_SIZE="$BATCH_SIZE"
    export MAX_FRAMES_NUM="$MAX_FRAMES_NUM"

    export USE_RUNTIME_CUT3R_GEOMETRY="False"
    export CHECK_SPATIAL_SIDECARS="True"
    export MODEL_USE_GEOMETRY_AWARE_PROJECTION="True"
    export MODEL_SPATIAL_ENCODER_TYPE="cut3r"
    export MODEL_SPATIAL_TOWER="cut3r_points"
    export MODEL_GEOMETRY_POSITION_MODE="$mode"
    export MODEL_NUM_GEOMETRY_PROJECTION_LAYERS="1"
    export MODEL_GEOMETRY_PROJECTION_NUM_HEADS="16"
    export MODEL_USE_AUXILIARY_GEOMETRY_HEAD="True"
    export MODEL_USE_AUXILIARY_GEOMETRY_LOSS="False"
    export MODEL_AUX_GEOMETRY_TARGETS="azimuth,elevation,log_distance"
    export MODEL_LAMBDA_GEO="0.1"
    export MODEL_GEOMETRY_LOSS_TYPE="smooth_l1"
    export MODEL_DETACH_GEOMETRY_TARGETS="True"
    export MODEL_GEOMETRY_GATE_INIT="0.0"
    export MODEL_USE_GEOMETRY_CONFIDENCE_MASK="True"
    export MODEL_ALLOW_MISSING_GEOMETRY_TARGETS="False"
    export MODEL_GEOMETRY_POSITION_MAX_ABS="10.0"
    export MODEL_GEOMETRY_FIXED_SCENE_SCALE="5.0"
    export MODEL_GEOMETRY_PROJECTION_DROPOUT="0.0"
    export MODEL_FUSION_BLOCK=""

    local cmd=(
      sbatch
      --job-name "$job_name"
      --time "$EVAL_TIME"
      --partition "$EVAL_PARTITION"
      --qos "$EVAL_QOS"
      --export=ALL
      "$EVAL_SCRIPT"
    )

    printf '[CMD]'
    printf ' %q' "${cmd[@]}"
    echo
    if [[ "$DRY_RUN" != "True" ]]; then
      "${cmd[@]}"
    fi
  )
}

submit_one "eval_rope_depth_cut3r_sidecars_41519216" "depth" "$RUN_DEPTH"
submit_one "eval_rope_spherical_cut3r_sidecars_41520134" "spherical" "$RUN_SPHERICAL"
submit_one "eval_rope_xyz_cut3r_sidecars_41519421" "xyz" "$RUN_XYZ"

echo "[DONE] Submitted all requested eval jobs."
