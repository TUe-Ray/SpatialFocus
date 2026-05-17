#!/bin/bash
set -euo pipefail

# Submit controlled VSiBench eval jobs for the Room Size depth-range probes.
# Defaults target the selected spherical GeoRoPE checkpoint from 40790070.

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_DIR/eval_geo_rope_fusion_cut3r.sh}"

PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/rope_spherical_100p_40790070}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"

OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/depth_range_probes}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features_points}"

MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_spherical_rope}"
MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
MODEL_SPATIAL_TOWER="${MODEL_SPATIAL_TOWER:-cut3r}"
MODEL_SPATIAL_FEATURE_DIM="${MODEL_SPATIAL_FEATURE_DIM:-768}"
MODEL_SPATIAL_TOWER_SELECT_FEATURE="${MODEL_SPATIAL_TOWER_SELECT_FEATURE:-all_tokens}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
LIMIT="${LIMIT:-0}"

EVAL_TIME="${EVAL_TIME:-04:00:00}"
EVAL_PARTITION="${EVAL_PARTITION:-boost_usr_prod}"
EVAL_QOS="${EVAL_QOS:-normal}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$REPO_DIR/logs/eval}"

SUBMIT_ORIGINAL="${SUBMIT_ORIGINAL:-True}"
SUBMIT_PROBE1="${SUBMIT_PROBE1:-True}"
SUBMIT_PROBE2="${SUBMIT_PROBE2:-True}"
PROBE2_DEPTHS="${PROBE2_DEPTHS:-15 20 30 50}"
DRY_RUN="${DRY_RUN:-True}"

bool_on() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|y|Y|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

submit_variant() {
  local run_name="$1"
  local max_depth="$2"
  local train_max_depth="$3"
  local eval_max_depth="$4"
  local ntk_scaling="$5"

  local job_name="$run_name"
  mkdir -p "$SLURM_LOG_DIR"

  echo
  echo "==== Variant: $run_name ===="
  echo "max_depth=$max_depth train_max_depth=${train_max_depth:-<unset>} eval_max_depth=${eval_max_depth:-<unset>} ntk=$ntk_scaling"

  if bool_on "$DRY_RUN"; then
    printf 'RUN_NAME=%q PRETRAINED_LOCAL=%q OUTPUT_PATH=%q MODEL_FUSION_BLOCK=%q MODEL_GEO_ROPE_FUSION_MODE=%q MODEL_GEO_ROPE_FUSION_MAX_DEPTH=%q MODEL_GEO_ROPE_FUSION_TRAIN_MAX_DEPTH=%q MODEL_GEO_ROPE_FUSION_EVAL_MAX_DEPTH=%q MODEL_GEO_ROPE_FUSION_NTK_SCALING=%q sbatch --job-name %q --time %q --partition %q --qos %q --export=ALL %q\n' \
      "$run_name" "$PRETRAINED_LOCAL" "$OUTPUT_PATH" "$MODEL_FUSION_BLOCK" "$MODEL_GEO_ROPE_FUSION_MODE" \
      "$max_depth" "$train_max_depth" "$eval_max_depth" "$ntk_scaling" "$job_name" "$EVAL_TIME" "$EVAL_PARTITION" "$EVAL_QOS" "$EVAL_SCRIPT"
    return 0
  fi

  RUN_NAME="$run_name" \
  PRETRAINED_LOCAL="$PRETRAINED_LOCAL" \
  MODEL_BASE_LOCAL="$MODEL_BASE_LOCAL" \
  SIGLIP_LOCAL="$SIGLIP_LOCAL" \
  OUTPUT_PATH="$OUTPUT_PATH" \
  SPATIAL_FEATURES_ROOT="$SPATIAL_FEATURES_ROOT" \
  SPATIAL_FEATURES_SUBDIR="$SPATIAL_FEATURES_SUBDIR" \
  CHECK_SPATIAL_SIDECARS=True \
  MODEL_SPATIAL_TOWER="$MODEL_SPATIAL_TOWER" \
  MODEL_SPATIAL_FEATURE_DIM="$MODEL_SPATIAL_FEATURE_DIM" \
  MODEL_SPATIAL_TOWER_SELECT_FEATURE="$MODEL_SPATIAL_TOWER_SELECT_FEATURE" \
  MODEL_FUSION_BLOCK="$MODEL_FUSION_BLOCK" \
  MODEL_GEO_ROPE_FUSION_MODE="$MODEL_GEO_ROPE_FUSION_MODE" \
  MODEL_GEO_ROPE_FUSION_MAX_DEPTH="$max_depth" \
  MODEL_GEO_ROPE_FUSION_TRAIN_MAX_DEPTH="$train_max_depth" \
  MODEL_GEO_ROPE_FUSION_EVAL_MAX_DEPTH="$eval_max_depth" \
  MODEL_GEO_ROPE_FUSION_NTK_SCALING="$ntk_scaling" \
  MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT" \
  NUM_PROCESSES="$NUM_PROCESSES" \
  BATCH_SIZE="$BATCH_SIZE" \
  MAX_FRAMES_NUM="$MAX_FRAMES_NUM" \
  LIMIT="$LIMIT" \
  sbatch \
    --job-name "$job_name" \
    --time "$EVAL_TIME" \
    --partition "$EVAL_PARTITION" \
    --qos "$EVAL_QOS" \
    --export=ALL \
    --output "$SLURM_LOG_DIR/%x_%j.out" \
    --error "$SLURM_LOG_DIR/%x_%j.err" \
    "$EVAL_SCRIPT"
}

cd "$REPO_DIR"
mkdir -p "$SLURM_LOG_DIR" "$OUTPUT_PATH"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] Missing eval script: $EVAL_SCRIPT"
  exit 1
fi
for path in "$PRETRAINED_LOCAL" "$MODEL_BASE_LOCAL" "$SIGLIP_LOCAL"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
done
if [[ ! -f "$PRETRAINED_LOCAL/config.json" ]]; then
  echo "[ERROR] Missing checkpoint config: $PRETRAINED_LOCAL/config.json"
  exit 1
fi

if bool_on "$SUBMIT_ORIGINAL"; then
  submit_variant "probe2_d2_spherical_original_md10" "10.0" "" "" "False"
fi

if bool_on "$SUBMIT_PROBE1"; then
  submit_variant "probe1_d2_spherical_linear_maxdepth20" "20.0" "" "" "False"
fi

if bool_on "$SUBMIT_PROBE2"; then
  for depth in $PROBE2_DEPTHS; do
    submit_variant "probe2_d2_spherical_evalmd${depth}_noNTK" "10.0" "10.0" "$depth" "False"
  done
  for depth in $PROBE2_DEPTHS; do
    submit_variant "probe2_d2_spherical_evalmd${depth}_NTK" "10.0" "10.0" "$depth" "True"
  done
fi
