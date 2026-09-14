#!/usr/bin/env bash
# One-video numerical equivalence smoke for the two loss-only VSI variants.
# Each new auxiliary candidate is compared directly with the retained full
# pre-SFT cache that produced its source Common-7 LogME values.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
BASE_MODEL="/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2"
SIGLIP_MODEL="/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384"
FORWARD_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1"
TARGET_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1"
CUT3R_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features"
SAMPLE_INDICES="/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
LOCAL_DATA="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
CACHE_ROOT="/home/shaoruei/probe_cache/pre_sft_logme_remaining6_equivalence"
OUTPUT_DIR="$REPO_ROOT/logs/pre_sft_logme_proxy_remaining_diagnostics_v1"
LOG_DIR="$OUTPUT_DIR/equivalence_logs"
FULL_FEATURES="siglip_output,fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
mkdir -p "$CACHE_ROOT" "$LOG_DIR"

extract_one() {
  local output_name="$1" label="$2" variant="$3" artifact="$4" subdirs="$5" loss="${6:-}"
  local output_root="$CACHE_ROOT/$output_name"
  local program=("$REPO_ROOT/scripts/probing/extract_depth_probe_features.py")
  if [[ -n "$loss" ]]; then
    program=("$REPO_ROOT/scripts/probing/extract_pre_sft_loss_only_attestation.py" --attestation-loss "$loss")
  fi
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run --no-capture-output -n "$ENV_NAME" python -u \
    "${program[@]}" \
    --model-loading-mode pre_sft_fusion \
    --pre-sft-fusion-variant "$variant" \
    --c1-calibration-json "$artifact" \
    --model-label "$label" \
    --model-path "$BASE_MODEL" \
    --siglip-path "$SIGLIP_MODEL" \
    --feature-levels "$FULL_FEATURES" \
    --output-root "$output_root" \
    --sample-indices "$SAMPLE_INDICES" \
    --train-data-json "$LOCAL_DATA" \
    --feature-root "$CUT3R_ROOT" \
    --spatial-features-subdir "$subdirs" \
    --forward-frames-root "$FORWARD_ROOT" \
    --probe-targets-root "$TARGET_ROOT" \
    --image-folder "$FORWARD_ROOT" \
    --video-folder "$FORWARD_ROOT" \
    --frames-upbound 32 \
    --device cuda:0 \
    --device-map auto \
    --dtype float16 \
    --cache-dtype float16 \
    --runtime-root "$output_root/runtime" \
    --pre-sft-gpu-weight-budget 4GiB \
    --pre-sft-cpu-offload-budget 45GiB \
    --limit-videos 1 \
    --assert-first-video \
    --resume \
    2>&1 | tee "$LOG_DIR/${output_name}.log"
}

BASELINE_C1="/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json"
SS012_C1="/home/shaoruei/probe_outputs/c1_additive_v1/official/spatialstack_add.json"
extract_one baseline_depth c1_vlm3r_depth_loss_attestation c1_vlm3r "$BASELINE_C1" spatial_features depth
extract_one ss_depth c1_ss012_pointmap_loss_attestation c1_ss_add "$SS012_C1" '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' pointmap

conda run --no-capture-output -n "$ENV_NAME" python -u \
  "$REPO_ROOT/scripts/probing/verify_pre_sft_loss_only_logme_equivalence.py" \
  --cache-root "$CACHE_ROOT" --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/verification.log"
