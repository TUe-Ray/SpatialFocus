#!/usr/bin/env bash
# Regenerate one controlled B/C/D/E/H pre-SFT full-policy cache, score its
# common-seven LogME, then recycle only the feature tensors created here.
set -euo pipefail

ID="${1:-}"
case "$ID" in B|C|D|E|H) ;; *) echo "Usage: $0 {B|C|D|E|H}" >&2; exit 2 ;; esac

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKTREE="${CONTROLLED_WORKTREE:-/tmp/spatialfocus_controlled_fusion_logme_f4e2259}"
EXPECTED_COMMIT="f4e2259451ecf12d4da9a87dce121639642a0524"
ENV_PYTHON="${VLM3R_PYTHON:-/home/shaoruei/miniconda3/envs/vlm3r/bin/python}"
CACHE_BASE="${CONTROLLED_LOGME_CACHE_BASE:-/home/shaoruei/probe_cache/controlled_fusion_logme_recache_v1}"
OUTPUT_DIR="${CONTROLLED_LOGME_OUTPUT_DIR:-$REPO_ROOT/logs/pre_sft_logme_proxy_controlled_fusion_v1}"
C1_ROOT="${CONTROLLED_C1_ROOT:-/home/shaoruei/probe_outputs/controlled_fusion_pre_sft_v3/c1}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
LOG="$OUTPUT_DIR/extraction_${ID}.log"

[[ -x "$ENV_PYTHON" && -e "$WORKTREE/.git" && -f "$C1_ROOT/artifact_manifest.json" ]] || {
  echo "Missing Python, controlled worktree, or locked C1 manifest" >&2; exit 1;
}
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "Controlled worktree is not the expected commit $EXPECTED_COMMIT" >&2; exit 1;
}

case "$ID" in
  B) LABEL="c1_controlled_b"; SOURCES="12"; LLM_LAYERS="" ;;
  C) LABEL="c1_controlled_c"; SOURCES="12"; LLM_LAYERS="0" ;;
  D) LABEL="c1_controlled_d"; SOURCES="12"; LLM_LAYERS="0" ;;
  E) LABEL="c1_controlled_e"; SOURCES="12,12,12"; LLM_LAYERS="0,1,2" ;;
  H) LABEL="c1_controlled_h"; SOURCES="12,12,12"; LLM_LAYERS="0,1,2" ;;
esac

CACHE_ROOT="$CACHE_BASE/$ID"
FEATURE_DIR="$CACHE_ROOT/features/$LABEL"
mkdir -p "$CACHE_ROOT" "$OUTPUT_DIR"

# This is the current mandatory full representation policy.  LogME scores its
# fixed common-seven subset only after all levels exist and provenance passes.
FULL_LEVELS="siglip_output,fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"

echo "[RUN] controlled=$ID commit=$EXPECTED_COMMIT cache=$CACHE_ROOT output=$OUTPUT_DIR" | tee -a "$LOG"
env CUDA_VISIBLE_DEVICES=0,1 MPLCONFIGDIR=/tmp/controlled_fusion_logme_mpl \
  "$ENV_PYTHON" -u "$WORKTREE/scripts/probing/extract_depth_probe_features.py" \
  --model-label "$LABEL" --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant "$LABEL" \
  --c1-calibration-json "$C1_ROOT/$ID/c1.json" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
  --feature-levels "$FULL_LEVELS" --sample-indices "$SAMPLE_INDICES" --output-root "$CACHE_ROOT" \
  --train-data-json "$WORKTREE/scripts/probing/scannet_depth_probe_local_data.yaml" \
  --feature-root "$FEATURE_ROOT" --spatial-features-subdir '12:spatial_features' \
  --spatialstack-cut3r-layers "$SOURCES" --spatialstack-llm-layers "$LLM_LAYERS" \
  --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
  --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --frames-upbound 32 \
  --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
  --runtime-root "$CACHE_ROOT/runtime/$LABEL" --pre-sft-gpu-weight-budget 4GiB --pre-sft-cpu-offload-budget 45GiB \
  --assert-first-video --resume 2>&1 | tee -a "$LOG"

env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/controlled_fusion_logme_mpl \
  "$ENV_PYTHON" -u "$REPO_ROOT/scripts/probing/run_controlled_fusion_common7_logme.py" \
  --candidate "$ID" --cache-base "$CACHE_BASE" --output-dir "$OUTPUT_DIR" \
  --c1-manifest "$C1_ROOT/artifact_manifest.json" --device cuda:0 --block-frames 8 2>&1 | tee -a "$LOG"

# This target is explicitly resolved, was created by this wrapper, and is
# safe to recycle only after a successful persisted LogME result.
case "$FEATURE_DIR" in "$CACHE_BASE"/[BCDEH]/features/c1_controlled_*) ;; *) echo "Unsafe cleanup target: $FEATURE_DIR" >&2; exit 1 ;; esac
[[ -d "$FEATURE_DIR" ]] || { echo "Expected feature directory is absent: $FEATURE_DIR" >&2; exit 1; }
rm -rf -- "$FEATURE_DIR"
echo "[COMPLETE] controlled=$ID; recycled generated features at $FEATURE_DIR" | tee -a "$LOG"
