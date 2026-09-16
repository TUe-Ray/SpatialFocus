#!/usr/bin/env bash
# Extract the official default-initialized A-prime pre-SFT representation and
# run cache-only Common-7 LogME.  Existing formal/17-model outputs are immutable.
set -euo pipefail

MODE="${1:-}"
if [[ ! "$MODE" =~ ^(preflight|smoke|full)$ ]]; then
  echo "Usage: $0 {preflight|smoke|full}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_PYTHON="${VLM3R_PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
BASE_MODEL="/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2"
SIGLIP_MODEL="/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384"
FORWARD_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1"
TARGET_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1"
FEATURE_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features"
SAMPLE_INDICES="/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
DATA_YAML="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
CACHE_ROOT="${A_PRIME_LOGME_CACHE_ROOT:-/home/shaoruei/probe_cache/pre_sft_logme_a_prime_common7}"
SMOKE_ROOT="${A_PRIME_LOGME_SMOKE_ROOT:-/home/shaoruei/probe_cache/pre_sft_logme_a_prime_smoke_v1}"
OUTPUT_DIR="${A_PRIME_LOGME_OUTPUT_DIR:-$REPO_ROOT/logs/pre_sft_logme_proxy_a_prime_common7}"
LABEL="pre_sft_controlled_a_prime"
FULL_LEVELS="siglip_output,fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"

require_inputs() {
  local path
  for path in "$ENV_PYTHON" "$BASE_MODEL/config.json" "$SIGLIP_MODEL/config.json" \
    "$SAMPLE_INDICES" "$DATA_YAML" "$FEATURE_ROOT/scannet/spatial_features/scene0384_00.pt"; do
    [[ -e "$path" ]] || { echo "Missing required A-prime input: $path" >&2; exit 1; }
  done
  for path in "$FORWARD_ROOT" "$TARGET_ROOT" "$FEATURE_ROOT"; do
    [[ -d "$path" ]] || { echo "Missing required A-prime directory: $path" >&2; exit 1; }
  done
  git -C "$REPO_ROOT" merge-base --is-ancestor ec0679e HEAD || {
    echo "Current HEAD does not contain the audited A-prime architecture commit ec0679e" >&2
    exit 1
  }
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "A-prime extraction requires a clean worktree for exact provenance" >&2
    exit 1
  }
  local forbidden
  forbidden="$(find "$BASE_MODEL" -type f \( -name adapter_model.bin -o -name non_lora_trainables.bin -o -name adapter_config.json \) -print -quit)"
  [[ -z "$forbidden" ]] || { echo "Base model contains forbidden post-SFT state: $forbidden" >&2; exit 1; }
}

extract() {
  local root="$1" label="$2"
  shift 2
  mkdir -p "$root" "$OUTPUT_DIR"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/a_prime_logme_mpl \
    "$ENV_PYTHON" -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$label" \
    --model-loading-mode pre_sft_fusion \
    --pre-sft-fusion-variant controlled_a_prime \
    --fusion-init-seed 42 \
    --common-model-init-seed 0 \
    --seed 42 \
    --model-path "$BASE_MODEL" \
    --siglip-path "$SIGLIP_MODEL" \
    --feature-levels "$FULL_LEVELS" \
    --output-root "$root" \
    --sample-indices "$SAMPLE_INDICES" \
    --train-data-json "$DATA_YAML" \
    --feature-root "$FEATURE_ROOT" \
    --spatial-features-subdir '12:spatial_features' \
    --spatialstack-cut3r-layers 12 \
    --spatialstack-llm-layers '' \
    --forward-frames-root "$FORWARD_ROOT" \
    --probe-targets-root "$TARGET_ROOT" \
    --image-folder "$FORWARD_ROOT" \
    --video-folder "$FORWARD_ROOT" \
    --frames-upbound 32 \
    --device cuda:0 \
    --device-map auto \
    --dtype float16 \
    --cache-dtype float16 \
    --runtime-root "$root/runtime" \
    --pre-sft-gpu-weight-budget 4GiB \
    --pre-sft-cpu-offload-budget 45GiB \
    --assert-first-video \
    --resume \
    "$@"
}

require_inputs
if [[ "$MODE" == preflight ]]; then
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$ENV_PYTHON" -m py_compile \
    "$REPO_ROOT/llava/model/controlled_fusion_pre_sft.py" \
    "$REPO_ROOT/scripts/diagnose_layerwise_spatial_hidden_scan.py" \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    "$REPO_ROOT/scripts/probing/run_pre_sft_a_prime_common7_logme.py"
  echo "[PASS] A-prime LogME preflight"
  exit 0
fi

if [[ "$MODE" == smoke ]]; then
  extract "$SMOKE_ROOT" "${LABEL}_smoke" --limit-videos 1
  echo "[PASS] A-prime one-video pre-SFT representation smoke"
  exit 0
fi

if [[ -f "$OUTPUT_DIR/logme_summary.json" ]] && jq -e '
  (.protocol.schema_version == "pre_sft_logme_proxy_a_prime_common7_v1")
  and (.scores | length == 1)
  and (.scores[0].architecture == "pre_sft_controlled_a_prime")
  and (.scores[0].status == "complete")
  and (.per_layer_rows == 7)
' "$OUTPUT_DIR/logme_summary.json" >/dev/null; then
  echo "[REUSE] complete A-prime Common-7 LogME result: $OUTPUT_DIR"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
extract "$CACHE_ROOT" "$LABEL" 2>&1 | tee "$OUTPUT_DIR/extraction.log"
env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/a_prime_logme_mpl \
  "$ENV_PYTHON" -u "$REPO_ROOT/scripts/probing/run_pre_sft_a_prime_common7_logme.py" \
  --cache-root "$CACHE_ROOT" --output-dir "$OUTPUT_DIR" --device cuda:0 --block-frames 8 \
  2>&1 | tee "$OUTPUT_DIR/logme.log"

jq -e '.scores[0].status == "complete" and .per_layer_rows == 7' "$OUTPUT_DIR/logme_summary.json" >/dev/null
FEATURE_DIR="$CACHE_ROOT/features/$LABEL"
for level in ${FULL_LEVELS//,/ }; do
  if [[ -d "$FEATURE_DIR/$level" ]]; then
    find "$FEATURE_DIR/$level" -type f -delete
    rmdir "$FEATURE_DIR/$level"
  fi
done
echo "[COMPLETE] A-prime Common-7 LogME saved; regeneratable feature tensors recycled"
