#!/usr/bin/env bash
# Complete the current 15-feature policy for historical pre-SFT probes.
set -euo pipefail

MODE="${1:-}"
if [[ ! "$MODE" =~ ^(preflight|smoke|full|summarize)$ ]]; then
  echo "Usage: $0 {preflight|smoke|full|summarize}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
CANDIDATES="${CANDIDATES:-SS012 SS123 SS036 SSCROSS VLM3R GEOROPE SELECTIVE}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
GEOMETRY_ROOT="${GEOMETRY_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_point_maps_32_v1}"
EOMT_ROOT="${EOMT_ROOT:-/home/shaoruei/probe_cache/eomt_consumer_grid_v2}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/legacy_pre_sft_completion_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/legacy_pre_sft_completion_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/legacy_pre_sft_completion_v1}"
ARTIFACT_MANIFEST="$DURABLE_ROOT/provenance/artifact_manifest.json"
SMOKE_MANIFEST="$DURABLE_ROOT/provenance/smoke_1train_1val.json"
SMOKE_MARKER="$DURABLE_ROOT/provenance/smoke_verification.json"
GPU_WEIGHT_BUDGET="${PRE_SFT_GPU_WEIGHT_BUDGET:-4GiB}"
CPU_OFFLOAD_BUDGET="${PRE_SFT_CPU_OFFLOAD_BUDGET:-45GiB}"
RECYCLE_FEATURE_CACHE="${RECYCLE_FEATURE_CACHE:-1}"

source "$REPO_ROOT/scripts/probing/common_probe_layers.sh"
mkdir -p "$CACHE_ROOT" "$DURABLE_ROOT/provenance" "$LOG_ROOT"

run() {
  printf '[COMMAND] '; printf '%q ' "$@"; printf '\n'
  "$@"
}

candidate_field() {
  local id="$1" field="$2"
  case "$id:$field" in
    BASE:label) printf 'pre_sft_base_vlm' ;;
    SS012:label) printf 'c1_spatialstack_add' ;; SS012:variant) printf 'c1_ss_add' ;; SS012:c1) printf '/home/shaoruei/probe_outputs/c1_additive_v1/official/spatialstack_add.json' ;; SS012:subdir) printf '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' ;; SS012:sources) printf '6,9,12' ;; SS012:layers) printf '0,1,2' ;;
    SS123:label) printf 'c1_spatialstack_add_123' ;; SS123:variant) printf 'c1_ss_add' ;; SS123:c1) printf '/home/shaoruei/probe_outputs/c1_ss_add_123/official/spatialstack_add.json' ;; SS123:subdir) printf '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' ;; SS123:sources) printf '6,9,12' ;; SS123:layers) printf '1,2,3' ;;
    SS036:label) printf 'c1_spatialstack_add_036' ;; SS036:variant) printf 'c1_ss_add' ;; SS036:c1) printf '/home/shaoruei/probe_outputs/c1_ss_add_036/official/spatialstack_add.json' ;; SS036:subdir) printf '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' ;; SS036:sources) printf '6,9,12' ;; SS036:layers) printf '0,3,6' ;;
    SSCROSS:label) printf 'c1_spatialstack_cross_attn_v1' ;; SSCROSS:variant) printf 'c1_ss_cross_attn_v1' ;; SSCROSS:c1) printf '/home/shaoruei/probe_outputs/c1_ss_cross_attn_v1/official/spatialstack_cross_attn_v1.json' ;; SSCROSS:subdir) printf '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' ;; SSCROSS:sources) printf '6,9,12' ;; SSCROSS:layers) printf '0,1,2' ;;
    VLM3R:label) printf 'c1_vlm3r' ;; VLM3R:variant) printf 'c1_vlm3r' ;; VLM3R:c1) printf '/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json' ;; VLM3R:subdir) printf 'spatial_features' ;; VLM3R:sources|VLM3R:layers) printf '' ;;
    GEOROPE:label) printf 'c1_geo_rope_fusion' ;; GEOROPE:variant) printf 'c1_geo_rope_fusion' ;; GEOROPE:c1) printf '/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json' ;; GEOROPE:subdir) printf 'spatial_features' ;; GEOROPE:sources|GEOROPE:layers) printf '' ;;
    SELECTIVE:label) printf 'c1_vlm3r_eomt_selective' ;; SELECTIVE:variant) printf 'c1_vlm3r' ;; SELECTIVE:c1) printf '/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json' ;; SELECTIVE:subdir) printf 'spatial_features' ;; SELECTIVE:sources|SELECTIVE:layers) printf '' ;;
    OBJECT:label) printf 'c1_eomt_object' ;; OBJECT:variant) printf 'c1_eomt_object' ;; OBJECT:c1) printf '/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json' ;; OBJECT:subdir) printf 'spatial_features' ;; OBJECT:sources|OBJECT:layers) printf '' ;;
    VISUAL:label) printf 'c1_visual_geo_rope' ;; VISUAL:variant) printf 'c1_visual_geo_rope' ;; VISUAL:c1) printf '/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json' ;; VISUAL:subdir) printf 'spatial_features' ;; VISUAL:sources|VISUAL:layers) printf '' ;;
    *) echo "Unsupported candidate/field: $id/$field" >&2; return 2 ;;
  esac
}

validate_candidates() {
  local id
  for id in $CANDIDATES; do
    [[ "$id" =~ ^(SS012|SS123|SS036|SSCROSS|VLM3R|GEOROPE|SELECTIVE|OBJECT|VISUAL)$ ]] || { echo "Unsupported CANDIDATES entry: $id" >&2; exit 2; }
  done
}

require_inputs() {
  local path id
  for path in "$BASE_MODEL/config.json" "$SIGLIP_MODEL/config.json" "$SAMPLE_INDICES" "$DATA_YAML" \
    "$FEATURE_ROOT/scannet/spatial_features/scene0384_00.pt" "$FEATURE_ROOT/scannet/spatial_features_dec_6/scene0384_00.pt" \
    "$FEATURE_ROOT/scannet/spatial_features_dec_9/scene0384_00.pt" "$FORWARD_ROOT" "$TARGET_ROOT"; do
    [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
  done
  for id in $CANDIDATES; do
    path="$(candidate_field "$id" c1)"; [[ -f "$path" ]] || { echo "Missing C1 artifact for $id: $path" >&2; exit 1; }
  done
  [[ -f "$GEOMETRY_ROOT/scannet/spatial_features_points/scene0000_00.pt" ]] || { echo "Missing full-frame ScanNet geometry sidecars" >&2; exit 1; }
  [[ -f "$EOMT_ROOT/validation.json" ]] || { echo "Missing EoMT validation cache" >&2; exit 1; }
  local forbidden
  forbidden="$(find "$BASE_MODEL" -type f \( -name adapter_model.bin -o -name non_lora_trainables.bin -o -name adapter_config.json \) -print -quit)"
  [[ -z "$forbidden" ]] || { echo "Forbidden post-SFT artifact in base model: $forbidden" >&2; exit 1; }
}

require_clean_commit() {
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "Formal completion requires a clean worktree" >&2; exit 1; }
}

lock_artifacts() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/prepare_legacy_pre_sft_completion.py" \
    --output "$ARTIFACT_MANIFEST" --reuse-existing --git-commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    --base-model "$BASE_MODEL" --siglip-model "$SIGLIP_MODEL" --sample-indices "$SAMPLE_INDICES" \
    --geometry-root "$GEOMETRY_ROOT" --eomt-root "$EOMT_ROOT" \
    --candidates "${CANDIDATES// /,}"
}

require_gpu() {
  local purpose="$1"
  local readiness="$DURABLE_ROOT/provenance/gpu_${GPU}_${purpose}_readiness.json"
  nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" --physical-gpu-id "$GPU" --output "$readiness"
}

preflight() {
  validate_candidates; require_inputs; require_clean_commit; lock_artifacts; require_gpu preflight
  run conda run -n "$ENV_NAME" python -m py_compile \
    "$REPO_ROOT/scripts/probing/legacy_pre_sft_completion_specs.py" \
    "$REPO_ROOT/scripts/probing/prepare_legacy_pre_sft_completion.py" \
    "$REPO_ROOT/scripts/probing/verify_legacy_pre_sft_completion_smoke.py" \
    "$REPO_ROOT/scripts/probing/summarize_legacy_pre_sft_completion.py"
  echo "[PASS] legacy pre-SFT completion preflight"
}

make_smoke_manifest() {
  [[ -f "$SMOKE_MANIFEST" ]] && return 0
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$SAMPLE_INDICES" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
}

extract_baseline() {
  local namespace="$1" manifest="$2" output log
  output="$CACHE_ROOT/$namespace/BASE"; log="$LOG_ROOT/${namespace}_BASE_extract.log"
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/legacy_presft_mpl conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label pre_sft_base_vlm --model-loading-mode pre_sft_base_vlm --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --feature-levels "$PRE_SFT_FULL_FEATURE_LEVELS_CSV" --sample-indices "$manifest" --output-root "$output" --train-data-json "$DATA_YAML" \
    --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" \
    --frames-upbound 32 --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
    --runtime-root "$output/runtime/pre_sft_base_vlm" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" \
    --assert-first-video --resume 2>&1 | tee "$log"
}

extract_candidate() {
  local namespace="$1" id="$2" manifest="$3" label variant c1 subdir sources layers output log
  label="$(candidate_field "$id" label)"; variant="$(candidate_field "$id" variant)"; c1="$(candidate_field "$id" c1)"
  subdir="$(candidate_field "$id" subdir)"; sources="$(candidate_field "$id" sources)"; layers="$(candidate_field "$id" layers)"
  output="$CACHE_ROOT/$namespace/$id"; log="$LOG_ROOT/${namespace}_${id}_extract.log"; mkdir -p "$output"
  local command=("$REPO_ROOT/scripts/probing/extract_depth_probe_features.py"
    --model-label "$label" --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant "$variant" --c1-calibration-json "$c1"
    --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --feature-levels "$PRE_SFT_FULL_FEATURE_LEVELS_CSV"
    --sample-indices "$manifest" --output-root "$output" --train-data-json "$DATA_YAML" --feature-root "$FEATURE_ROOT"
    --spatial-features-subdir "$subdir" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT"
    --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --frames-upbound 32 --device cuda:0 --device-map auto
    --dtype float16 --cache-dtype float16 --runtime-root "$output/runtime/$label" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET"
    --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" --assert-first-video --resume)
  [[ -z "$sources" ]] || command+=(--spatialstack-cut3r-layers "$sources" --spatialstack-llm-layers "$layers")
  if [[ "$id" == GEOROPE || "$id" == VISUAL ]]; then
    local geometry_architecture
    [[ "$id" == GEOROPE ]] && geometry_architecture=geo_rope_fusion || geometry_architecture=visual_geo_rope
    command+=(--geometry-c1-calibration-json "/home/shaoruei/probe_outputs/c1_geometry_pre_sft_v1/$geometry_architecture/c1_activation.json"
      --geometry-spatial-features-root "$GEOMETRY_ROOT" --geometry-spatial-features-subdir spatial_features_points --geometry-point-map-key point_maps_ref)
  elif [[ "$id" == SELECTIVE ]]; then
    command+=(--eomt-selective-kv-gate --eomt-consumer-cache-root "$EOMT_ROOT" --eomt-cache-validation "$EOMT_ROOT/validation.json")
    [[ "$namespace" == smoke ]] && command+=(--verify-eomt-file-checksum)
  elif [[ "$id" == OBJECT ]]; then
    command+=(--eomt-consumer-cache-root "$EOMT_ROOT" --eomt-cache-validation "$EOMT_ROOT/validation.json")
    [[ "$namespace" == smoke ]] && command+=(--verify-eomt-file-checksum)
  fi
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/legacy_presft_mpl conda run -n "$ENV_NAME" python -u "${command[@]}" 2>&1 | tee "$log"
}

train_level() {
  local root="$1" id="$2" level="$3" gpu="$4" epochs="$5" manifest="$6" label
  label="$(candidate_field "$id" label)"
  env CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/legacy_presft_mpl conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$root" --sample-indices "$manifest" --probe-subdir probes --model-labels "$label" --feature-levels "$level" \
    --epochs "$epochs" --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 --probe-seed 0 \
    --experiment-variant "legacy_${id}_pre_sft_completion" --device cuda:0 --no-write-aggregate --skip-existing \
    2>&1 | tee "$LOG_ROOT/${id}_${level}_probe.log"
}

train_candidate() {
  local namespace="$1" id="$2" epochs="$3" manifest="$4" root levels index first second first_pid second_pid
  root="$CACHE_ROOT/$namespace/$id"
  IFS=',' read -r -a levels <<< "$PRE_SFT_FULL_FEATURE_LEVELS_CSV"; index=0
  while [[ "$index" -lt "${#levels[@]}" ]]; do
    first="${levels[$index]}"; train_level "$root" "$id" "$first" 0 "$epochs" "$manifest" & first_pid=$!; index=$((index + 1))
    if [[ "$index" -lt "${#levels[@]}" ]]; then
      second="${levels[$index]}"; train_level "$root" "$id" "$second" 1 "$epochs" "$manifest" & second_pid=$!
      wait "$first_pid"; wait "$second_pid"; index=$((index + 1))
    else
      wait "$first_pid"
    fi
  done
}

smoke() {
  preflight; make_smoke_manifest; extract_baseline smoke "$SMOKE_MANIFEST"; train_candidate smoke BASE 2 "$SMOKE_MANIFEST"
  local id
  for id in $CANDIDATES; do extract_candidate smoke "$id" "$SMOKE_MANIFEST"; train_candidate smoke "$id" 2 "$SMOKE_MANIFEST"; done
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_legacy_pre_sft_completion_smoke.py" \
    --cache-root "$CACHE_ROOT/smoke" --artifact-manifest "$ARTIFACT_MANIFEST" --sample-indices "$SMOKE_MANIFEST" --output "$SMOKE_MARKER"
}

preserve_results() {
  local id="$1" label root destination
  label="$(candidate_field "$id" label)"; root="$CACHE_ROOT/full/$id"; destination="$DURABLE_ROOT/results/$id"
  [[ ! -e "$destination" ]] || { echo "Refusing to overwrite durable result: $destination" >&2; exit 1; }
  mkdir -p "$destination"; cp -a "$root/probes/$label" "$destination/probes"; cp -a "$root/features/$label/extraction_provenance.json" "$destination/extraction_provenance.json"
}

recycle_features() {
  local id="$1" label target
  label="$(candidate_field "$id" label)"; target="$CACHE_ROOT/full/$id/features/$label"
  [[ "$RECYCLE_FEATURE_CACHE" == 1 ]] || return 0
  case "$target" in "$CACHE_ROOT"/full/SS012/features/c1_spatialstack_add|"$CACHE_ROOT"/full/SS123/features/c1_spatialstack_add_123|"$CACHE_ROOT"/full/SS036/features/c1_spatialstack_add_036|"$CACHE_ROOT"/full/SSCROSS/features/c1_spatialstack_cross_attn_v1|"$CACHE_ROOT"/full/VLM3R/features/c1_vlm3r|"$CACHE_ROOT"/full/GEOROPE/features/c1_geo_rope_fusion|"$CACHE_ROOT"/full/SELECTIVE/features/c1_vlm3r_eomt_selective|"$CACHE_ROOT"/full/OBJECT/features/c1_eomt_object|"$CACHE_ROOT"/full/VISUAL/features/c1_visual_geo_rope) ;; *) echo "Refusing unexpected cleanup target: $target" >&2; exit 1;; esac
  [[ -d "$target" ]] || return 0
  echo "[RECYCLE] removing regenerated feature tensors after durable result preservation: $target"; rm -rf -- "$target"
}

full() {
  [[ -f "$SMOKE_MARKER" ]] || { echo "Run '$0 smoke' successfully first: $SMOKE_MARKER" >&2; exit 1; }
  require_inputs; require_clean_commit; lock_artifacts; require_gpu full
  local id
  for id in $CANDIDATES; do extract_candidate full "$id" "$SAMPLE_INDICES"; train_candidate full "$id" 50 "$SAMPLE_INDICES"; preserve_results "$id"; recycle_features "$id"; done
  summarize
}

summarize() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_legacy_pre_sft_completion.py" \
    --results-root "$DURABLE_ROOT/results" --artifact-manifest "$ARTIFACT_MANIFEST" --sample-indices "$SAMPLE_INDICES" --output-dir "$DURABLE_ROOT/summary"
}

case "$MODE" in
  preflight) preflight ;;
  smoke) smoke ;;
  full) full ;;
  summarize) summarize ;;
esac
