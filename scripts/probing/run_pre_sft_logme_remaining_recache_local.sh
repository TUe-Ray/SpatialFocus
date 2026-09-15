#!/usr/bin/env bash
# Rebuild one missing non-formal pre-SFT representation, run Common-7 LogME,
# and recycle only the feature tensors created by this wrapper after success.
set -euo pipefail

KEY="${1:-}"
if [[ ! "$KEY" =~ ^(ss_depth|extra_object_token|visual_geo_rope)$ ]]; then
  echo "Usage: $0 {ss_depth|extra_object_token|visual_geo_rope}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
export PATH="/home/shaoruei/miniconda3/bin:${PATH:-}"
if [[ -f /home/shaoruei/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/shaoruei/miniconda3/etc/profile.d/conda.sh
fi
ENV_NAME="${ENV_NAME:-vlm3r}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
BASE_MODEL="/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2"
SIGLIP_MODEL="/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384"
FORWARD_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1"
TARGET_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1"
CUT3R_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features"
POINTMAP_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/cut3r_point_maps_32_v1"
SAMPLE_INDICES="/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
C1_ARTIFACT="/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json"
LOCAL_DATA="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
CACHE_BASE="/home/shaoruei/probe_cache/pre_sft_logme_remaining6_recache"
OUTPUT_DIR="$REPO_ROOT/logs/pre_sft_logme_proxy_remaining_diagnostics_v1"
LOG_DIR="$OUTPUT_DIR/extraction_logs"
FULL_FEATURES="siglip_output,fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
PROGRAM=("$REPO_ROOT/scripts/probing/extract_depth_probe_features.py")
SPATIAL_SUBDIR="spatial_features"

if [[ "$KEY" == "ss_depth" ]]; then
  VARIANT="c1_ss_add"
  LABEL="ss_depth"
  FULL_ROOT="$CACHE_BASE/ss_depth"
  C1_ARTIFACT="/home/shaoruei/probe_outputs/c1_additive_v1/official/spatialstack_add.json"
  SPATIAL_SUBDIR="6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features"
  PROGRAM=("$REPO_ROOT/scripts/probing/extract_pre_sft_loss_only_attestation.py" --attestation-loss pointmap)
  EXTRA_ARGS=()
elif [[ "$KEY" == "extra_object_token" ]]; then
  VARIANT="c1_eomt_object"
  LABEL="c1_eomt_object"
  FULL_ROOT="$CACHE_BASE/eomt_object"
  EOMT_ROOT="/home/shaoruei/probe_cache/eomt_consumer_grid_v2"
  EXTRA_ARGS=(
    --eomt-consumer-cache-root "$EOMT_ROOT"
    --eomt-cache-validation "$EOMT_ROOT/validation.json"
  )
else
  VARIANT="c1_visual_geo_rope"
  LABEL="c1_visual_geo_rope"
  FULL_ROOT="$CACHE_BASE/visual_geo_rope"
  ACTIVATION="/home/shaoruei/probe_outputs/c1_geometry_pre_sft_v1/visual_geo_rope/c1_activation.json"
  EXTRA_ARGS=(
    --geometry-c1-calibration-json "$ACTIVATION"
    --geometry-spatial-features-root "$POINTMAP_ROOT"
    --geometry-spatial-features-subdir spatial_features_points
    --geometry-point-map-key point_maps_ref
  )
fi

for path in "$BASE_MODEL/config.json" "$SIGLIP_MODEL/config.json" "$SAMPLE_INDICES" "$C1_ARTIFACT" "$LOCAL_DATA"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
for path in "$FORWARD_ROOT" "$TARGET_ROOT" "$CUT3R_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 1; }
done
if [[ "$KEY" == "extra_object_token" ]]; then
  [[ -f "$EOMT_ROOT/validation.json" ]] || { echo "Missing EoMT validation: $EOMT_ROOT/validation.json" >&2; exit 1; }
elif [[ "$KEY" == "visual_geo_rope" ]]; then
  [[ -f "$ACTIVATION" && -d "$POINTMAP_ROOT" ]] || { echo "Missing Visual geo-RoPE activation or point maps" >&2; exit 1; }
fi

mkdir -p "$FULL_ROOT" "$LOG_DIR"
RESULT_COMPLETE=false
if [[ -f "$OUTPUT_DIR/source_provenance_${KEY}.json" && -f "$OUTPUT_DIR/logme_summary.json" ]] && \
  jq -e --arg model_label "$LABEL" '
    (.protocol.schema_version == "pre_sft_logme_proxy_remaining_diagnostics_v1")
    and (.protocol.primary_layers == [1, 3, 6, 9, 15, 21, 27])
    and (.protocol.dtype == "float64")
    and (.protocol.training_videos == 1006)
    and (.protocol.training_frames == 2012)
    and (.protocol.formal_target_signature == "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37")
    and (([.scores[] | select(
      .architecture == $model_label
      and .common7_mean_logme != null
      and .valid_tokens_per_layer == 394352
      and .training_videos == 1006
      and .target_signature == "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37"
    )] | length) == 1)
  ' "$OUTPUT_DIR/logme_summary.json" >/dev/null; then
  RESULT_COMPLETE=true
  echo "[REUSE] complete provenance-verified Common-7 result: $KEY/$LABEL"
fi

if [[ "$RESULT_COMPLETE" != true ]]; then
  echo "[RUN] key=$KEY GPUs=$CUDA_DEVICES cache=$FULL_ROOT output=$OUTPUT_DIR"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run --no-capture-output -n "$ENV_NAME" python -u \
    "${PROGRAM[@]}" \
    --model-loading-mode pre_sft_fusion \
    --pre-sft-fusion-variant "$VARIANT" \
    --c1-calibration-json "$C1_ARTIFACT" \
    --model-label "$LABEL" \
    --model-path "$BASE_MODEL" \
    --siglip-path "$SIGLIP_MODEL" \
    --feature-levels "$FULL_FEATURES" \
    --output-root "$FULL_ROOT" \
    --sample-indices "$SAMPLE_INDICES" \
    --train-data-json "$LOCAL_DATA" \
    --feature-root "$CUT3R_ROOT" \
    --spatial-features-subdir "$SPATIAL_SUBDIR" \
    --forward-frames-root "$FORWARD_ROOT" \
    --probe-targets-root "$TARGET_ROOT" \
    --image-folder "$FORWARD_ROOT" \
    --video-folder "$FORWARD_ROOT" \
    --frames-upbound 32 \
    --device cuda:0 \
    --device-map auto \
    --dtype float16 \
    --cache-dtype float16 \
    --runtime-root "$FULL_ROOT/runtime" \
    --pre-sft-gpu-weight-budget 4GiB \
    --pre-sft-cpu-offload-budget 45GiB \
    --assert-first-video \
    --resume \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_DIR/${KEY}_extraction.log"

  env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/run_pre_sft_logme_remaining_diagnostics.py" \
    --candidate "$KEY" --output-dir "$OUTPUT_DIR" --device cuda:0 --block-frames 8 \
    2>&1 | tee "$LOG_DIR/${KEY}_logme.log"
fi

test -f "$OUTPUT_DIR/source_provenance_${KEY}.json"
jq -e --arg model_label "$LABEL" '
  ([.scores[] | select(.architecture == $model_label)]) as $rows
  | ($rows | length) == 1
    and ($rows[0].common7_mean_logme != null)
    and ($rows[0].valid_tokens_per_layer == 394352)
    and ($rows[0].training_videos == 1006)
' "$OUTPUT_DIR/logme_summary.json" >/dev/null

# The exact target is constrained to this wrapper-owned, regeneratable cache.
# Preserve targets/runtime until final audit but release the large tensors.
FEATURE_DIR="$FULL_ROOT/features/$LABEL"
if [[ -d "$FEATURE_DIR" ]]; then
  find "$FEATURE_DIR" -mindepth 1 -delete
fi
echo "[COMPLETE] $KEY LogME saved; regenerated feature tensors recycled from $FEATURE_DIR"
