#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_DIR/eval_posthoc_geo_rope_cut3r.sh}"

PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/archived_eomt/selec_100%_baseline_40390735}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"

OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/posthoc_geo_rope_cut3r}"
LIMIT="${LIMIT:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

EVAL_TIME="${EVAL_TIME:-04:00:00}"
EVAL_PARTITION="${EVAL_PARTITION:-boost_usr_prod}"
EVAL_QOS="${EVAL_QOS:-normal}"
DRY_RUN="${DRY_RUN:-False}"

MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
LAMBDAS="${LAMBDAS:-0.0 0.25 0.5 1.0}"

mkdir -p "$REPO_DIR/logs/eval" "$OUTPUT_PATH"

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
if [[ ! -f "$PRETRAINED_LOCAL/adapter_config.json" ]]; then
  echo "[ERROR] Missing LoRA adapter config: $PRETRAINED_LOCAL/adapter_config.json"
  exit 1
fi
if [[ ! -f "$PRETRAINED_LOCAL/non_lora_trainables.bin" ]]; then
  echo "[ERROR] Missing non-LoRA trainables: $PRETRAINED_LOCAL/non_lora_trainables.bin"
  echo "[ERROR] Use the real checkpoint directory, not a stale runtime directory with broken symlinks."
  exit 1
fi
if [[ ! -f "$PRETRAINED_LOCAL/adapter_model.bin" && ! -f "$PRETRAINED_LOCAL/adapter_model.safetensors" ]]; then
  echo "[ERROR] Missing LoRA adapter weights: expected adapter_model.bin or adapter_model.safetensors under $PRETRAINED_LOCAL"
  exit 1
fi

submit_one() {
  local lambda="$1"
  local lambda_tag
  lambda_tag="$(printf '%s' "$lambda" | tr '.' 'p')"
  local run_name="posthoc_geo_rope_${MODEL_GEO_ROPE_FUSION_MODE}_lambda_${lambda_tag}"
  local job_name="Eval_PostHocGeoRoPE_l${lambda_tag}"
  local fusion_block="svf_patch_only_geo_rope_eval"

  echo "==== Submit lambda=$lambda ===="
  echo "checkpoint=$PRETRAINED_LOCAL"
  echo "fusion_block=$fusion_block"
  echo "mode=$MODEL_GEO_ROPE_FUSION_MODE"
  echo "run_name=$run_name"
  echo "output_path=$OUTPUT_PATH"

  (
    export PRETRAINED_LOCAL="$PRETRAINED_LOCAL"
    export MODEL_BASE_LOCAL="$MODEL_BASE_LOCAL"
    export SIGLIP_LOCAL="$SIGLIP_LOCAL"
    export RUN_NAME="$run_name"
    export OUTPUT_PATH="$OUTPUT_PATH"
    export LIMIT="$LIMIT"
    export NUM_PROCESSES="$NUM_PROCESSES"
    export BATCH_SIZE="$BATCH_SIZE"
    export MAX_FRAMES_NUM="$MAX_FRAMES_NUM"

    export CHECK_SPATIAL_SIDECARS="True"
    export MODEL_SPATIAL_TOWER="cut3r"
    export MODEL_SPATIAL_FEATURE_DIM="768"
    export MODEL_SPATIAL_TOWER_SELECT_FEATURE="all_tokens"
    export MODEL_FUSION_BLOCK="$fusion_block"
    export MODEL_GEO_ROPE_FUSION_MODE="$MODEL_GEO_ROPE_FUSION_MODE"
    export MODEL_GEO_ROPE_FUSION_MAX_DEPTH="$MODEL_GEO_ROPE_FUSION_MAX_DEPTH"
    export MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
    export MODEL_GEO_ROPE_FUSION_LOG_STATS="$MODEL_GEO_ROPE_FUSION_LOG_STATS"
    export MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA="$lambda"

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

for lambda in $LAMBDAS; do
  submit_one "$lambda"
done

echo "[DONE] Submitted post-hoc GeoRoPE eval jobs."
