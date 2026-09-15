#!/usr/bin/env bash
# One-A100 Snellius executor for the seven legacy pre-SFT representation probes.
set -euo pipefail

MODE="${1:-}"
if [[ ! "$MODE" =~ ^(preflight|smoke|candidate|summarize)$ ]]; then
  echo "Usage: $0 {preflight|smoke|candidate|summarize} [CANDIDATE]" >&2
  exit 2
fi

REPO_ROOT="${SPATIALFOCUS_ROOT:-/home/shuang/SpatialFocus}"
PYTHON="${VLM3R_PYTHON:-/home/shuang/miniconda3/envs/vlm3r_a100_proxy/bin/python}"
DATA_ROOT="${PROBE_DATA_ROOT:-/scratch-shared/shuang/spatialfocus_probe_data}"
BASE_MODEL="${BASE_MODEL:-/home/shuang/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/home/shuang/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-$DATA_ROOT/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-$DATA_ROOT/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/cut3r_features}"
GEOMETRY_ROOT="${GEOMETRY_ROOT:-$DATA_ROOT/cut3r_point_maps_32_v1}"
EOMT_ROOT="${EOMT_ROOT:-$DATA_ROOT/eomt_consumer_grid_v2}"
C1_ROOT="${C1_ROOT:-/home/shuang/c1_artifacts}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shuang/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_snellius_data.yaml}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/shuang/legacy_pre_sft_completion_a100_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shuang/proxy_outputs/legacy_pre_sft_completion_a100_v1}"
LOG_ROOT="${LOG_ROOT:-$DURABLE_ROOT/logs}"
ARTIFACT_MANIFEST="$DURABLE_ROOT/provenance/artifact_manifest.json"
SMOKE_MANIFEST="$DURABLE_ROOT/provenance/smoke_1train_1val.json"
SMOKE_MARKER="$DURABLE_ROOT/provenance/smoke_verification.json"
CANDIDATES="SS012 SS123 SS036 SSCROSS VLM3R GEOROPE SELECTIVE"
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
    SS012:label) printf 'c1_spatialstack_add' ;; SS012:variant) printf 'c1_ss_add' ;; SS012:c1) printf '%s/c1_additive_v1/official/spatialstack_add.json' "$C1_ROOT" ;; SS012:subdir) printf '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' ;; SS012:sources) printf '6,9,12' ;; SS012:layers) printf '0,1,2' ;;
    SS123:label) printf 'c1_spatialstack_add_123' ;; SS123:variant) printf 'c1_ss_add' ;; SS123:c1) printf '%s/c1_ss_add_123/official/spatialstack_add.json' "$C1_ROOT" ;; SS123:subdir) printf '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' ;; SS123:sources) printf '6,9,12' ;; SS123:layers) printf '1,2,3' ;;
    SS036:label) printf 'c1_spatialstack_add_036' ;; SS036:variant) printf 'c1_ss_add' ;; SS036:c1) printf '%s/c1_ss_add_036/official/spatialstack_add.json' "$C1_ROOT" ;; SS036:subdir) printf '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' ;; SS036:sources) printf '6,9,12' ;; SS036:layers) printf '0,3,6' ;;
    SSCROSS:label) printf 'c1_spatialstack_cross_attn_v1' ;; SSCROSS:variant) printf 'c1_ss_cross_attn_v1' ;; SSCROSS:c1) printf '%s/c1_ss_cross_attn_v1/official/spatialstack_cross_attn_v1.json' "$C1_ROOT" ;; SSCROSS:subdir) printf '6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features' ;; SSCROSS:sources) printf '6,9,12' ;; SSCROSS:layers) printf '0,1,2' ;;
    VLM3R:label) printf 'c1_vlm3r' ;; VLM3R:variant) printf 'c1_vlm3r' ;; VLM3R:c1) printf '%s/c1_vlm3r_v1/official/vlm3r.json' "$C1_ROOT" ;; VLM3R:subdir) printf 'spatial_features' ;; VLM3R:sources|VLM3R:layers) printf '' ;;
    GEOROPE:label) printf 'c1_geo_rope_fusion' ;; GEOROPE:variant) printf 'c1_geo_rope_fusion' ;; GEOROPE:c1) printf '%s/c1_vlm3r_v1/official/vlm3r.json' "$C1_ROOT" ;; GEOROPE:subdir) printf 'spatial_features' ;; GEOROPE:sources|GEOROPE:layers) printf '' ;;
    SELECTIVE:label) printf 'c1_vlm3r_eomt_selective' ;; SELECTIVE:variant) printf 'c1_vlm3r' ;; SELECTIVE:c1) printf '%s/c1_vlm3r_v1/official/vlm3r.json' "$C1_ROOT" ;; SELECTIVE:subdir) printf 'spatial_features' ;; SELECTIVE:sources|SELECTIVE:layers) printf '' ;;
    *) echo "Unsupported candidate/field: $id/$field" >&2; return 2 ;;
  esac
}

require_clean_commit() {
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "Formal completion requires a clean worktree" >&2; exit 1; }
}

require_count() {
  local directory="$1" expected="$2" actual
  [[ -d "$directory" ]] || { echo "Missing input directory: $directory" >&2; exit 1; }
  actual="$(find "$directory" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  [[ "$actual" -eq "$expected" ]] || { echo "Expected $expected .pt inputs at $directory, found $actual" >&2; exit 1; }
}

require_inputs() {
  local id c1 forbidden
  [[ -x "$PYTHON" ]] || { echo "Missing proxy Python: $PYTHON" >&2; exit 1; }
  for c1 in "$BASE_MODEL/config.json" "$SIGLIP_MODEL/config.json" "$SAMPLE_INDICES" "$DATA_YAML"; do
    [[ -f "$c1" ]] || { echo "Missing required input: $c1" >&2; exit 1; }
  done
  require_count "$FORWARD_ROOT/frames/scannet" 1199
  require_count "$TARGET_ROOT/targets/scannet/spatial_features_points" 1199
  require_count "$FEATURE_ROOT/scannet/spatial_features" 1199
  require_count "$FEATURE_ROOT/scannet/spatial_features_dec_6" 1199
  require_count "$FEATURE_ROOT/scannet/spatial_features_dec_9" 1199
  require_count "$GEOMETRY_ROOT/scannet/spatial_features_points" 1199
  require_count "$EOMT_ROOT/class_logits/scannet" 1199
  require_count "$EOMT_ROOT/object_masks/scannet" 1199
  require_count "$EOMT_ROOT/selective_masks/scannet" 1199
  [[ -f "$EOMT_ROOT/validation.json" && -f "$EOMT_ROOT/checksums.json" ]] || { echo "Missing EoMT validation/checksum manifest" >&2; exit 1; }
  for id in $CANDIDATES; do
    c1="$(candidate_field "$id" c1)"
    [[ -f "$c1" ]] || { echo "Missing C1 artifact for $id: $c1" >&2; exit 1; }
  done
  [[ -f "$C1_ROOT/c1_geometry_pre_sft_v1/geo_rope_fusion/c1_activation.json" ]] || { echo "Missing GeoRoPE C1 activation" >&2; exit 1; }
  forbidden="$(find "$BASE_MODEL" -type f \( -name adapter_model.bin -o -name non_lora_trainables.bin -o -name adapter_config.json \) -print -quit)"
  [[ -z "$forbidden" ]] || { echo "Forbidden post-SFT artifact in base model: $forbidden" >&2; exit 1; }
}

lock_artifacts() {
  run "$PYTHON" -u "$REPO_ROOT/scripts/probing/prepare_legacy_pre_sft_completion.py" \
    --output "$ARTIFACT_MANIFEST" --reuse-existing --git-commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    --base-model "$BASE_MODEL" --siglip-model "$SIGLIP_MODEL" --sample-indices "$SAMPLE_INDICES" \
    --geometry-root "$GEOMETRY_ROOT" --eomt-root "$EOMT_ROOT" --c1-root "$C1_ROOT"
}

preflight() {
  require_clean_commit
  require_inputs
  lock_artifacts
  run "$PYTHON" -m py_compile \
    "$REPO_ROOT/scripts/probing/legacy_pre_sft_completion_specs.py" \
    "$REPO_ROOT/scripts/probing/prepare_legacy_pre_sft_completion.py" \
    "$REPO_ROOT/scripts/probing/verify_legacy_pre_sft_completion_smoke.py" \
    "$REPO_ROOT/scripts/probing/summarize_legacy_pre_sft_completion.py"
  "$PYTHON" - <<'PY'
import accelerate, peft, torch, transformers
assert peft.__version__ == "0.4.0", peft.__version__
assert torch.__version__.startswith("2.1.1"), torch.__version__
print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__})
PY
  echo "[PASS] Snellius legacy pre-SFT completion preflight"
}

make_smoke_manifest() {
  [[ -f "$SMOKE_MANIFEST" ]] && return 0
  run "$PYTHON" -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$SAMPLE_INDICES" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
}

extract_one() {
  local namespace="$1" id="$2" manifest="$3" label output log
  label="$(candidate_field "$id" label)"
  output="$CACHE_ROOT/$namespace/$id"
  log="$LOG_ROOT/${namespace}_${id}_extract.log"
  mkdir -p "$output"
  local command=("$PYTHON" -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py"
    --model-label "$label" --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL"
    --feature-levels "$PRE_SFT_FULL_FEATURE_LEVELS_CSV" --sample-indices "$manifest" --output-root "$output"
    --train-data-json "$DATA_YAML" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT"
    --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --frames-upbound 32
    --device cuda:0 --device-map cuda:0 --dtype float16 --cache-dtype float16
    --runtime-root "$output/runtime/$label" --assert-first-video --resume)
  if [[ "$id" == BASE ]]; then
    command+=(--model-loading-mode pre_sft_base_vlm)
  else
    local variant c1 subdir sources layers
    variant="$(candidate_field "$id" variant)"; c1="$(candidate_field "$id" c1)"
    subdir="$(candidate_field "$id" subdir)"; sources="$(candidate_field "$id" sources)"; layers="$(candidate_field "$id" layers)"
    command+=(--model-loading-mode pre_sft_fusion --pre-sft-fusion-variant "$variant" --c1-calibration-json "$c1"
      --feature-root "$FEATURE_ROOT" --spatial-features-subdir "$subdir")
    [[ -z "$sources" ]] || command+=(--spatialstack-cut3r-layers "$sources" --spatialstack-llm-layers "$layers")
    if [[ "$id" == GEOROPE ]]; then
      command+=(--geometry-c1-calibration-json "$C1_ROOT/c1_geometry_pre_sft_v1/geo_rope_fusion/c1_activation.json"
        --geometry-spatial-features-root "$GEOMETRY_ROOT" --geometry-spatial-features-subdir spatial_features_points
        --geometry-point-map-key point_maps_ref)
    elif [[ "$id" == SELECTIVE ]]; then
      command+=(--eomt-selective-kv-gate --eomt-consumer-cache-root "$EOMT_ROOT"
        --eomt-cache-validation "$EOMT_ROOT/validation.json")
      [[ "$namespace" == smoke ]] && command+=(--verify-eomt-file-checksum)
    fi
  fi
  "${command[@]}" 2>&1 | tee "$log"
}

train_candidate() {
  local namespace="$1" id="$2" epochs="$3" manifest="$4" root label level
  root="$CACHE_ROOT/$namespace/$id"; label="$(candidate_field "$id" label)"
  IFS=',' read -r -a levels <<< "$PRE_SFT_FULL_FEATURE_LEVELS_CSV"
  for level in "${levels[@]}"; do
    "$PYTHON" -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
      --output-root "$root" --sample-indices "$manifest" --probe-subdir probes --model-labels "$label" \
      --feature-levels "$level" --epochs "$epochs" --batch-size 32 --lr 1e-3 --early-stop-patience 10 \
      --num-workers 0 --probe-seed 0 --experiment-variant "legacy_${id}_pre_sft_completion_a100" \
      --device cuda:0 --no-write-aggregate --skip-existing 2>&1 | tee "$LOG_ROOT/${namespace}_${id}_${level}_probe.log"
  done
}

smoke() {
  preflight
  make_smoke_manifest
  local id
  extract_one smoke BASE "$SMOKE_MANIFEST"
  train_candidate smoke BASE 2 "$SMOKE_MANIFEST"
  for id in $CANDIDATES; do
    extract_one smoke "$id" "$SMOKE_MANIFEST"
    train_candidate smoke "$id" 2 "$SMOKE_MANIFEST"
  done
  run "$PYTHON" -u "$REPO_ROOT/scripts/probing/verify_legacy_pre_sft_completion_smoke.py" \
    --cache-root "$CACHE_ROOT/smoke" --artifact-manifest "$ARTIFACT_MANIFEST" \
    --sample-indices "$SMOKE_MANIFEST" --output "$SMOKE_MARKER"
}

preserve_and_recycle() {
  local id="$1" label root destination feature_target
  label="$(candidate_field "$id" label)"; root="$CACHE_ROOT/full/$id"; destination="$DURABLE_ROOT/results/$id"
  [[ ! -e "$destination" ]] || { echo "Refusing to overwrite durable result: $destination" >&2; exit 1; }
  mkdir -p "$destination"
  cp -a "$root/probes/$label" "$destination/probes"
  cp -a "$root/features/$label/extraction_provenance.json" "$destination/extraction_provenance.json"
  [[ "$RECYCLE_FEATURE_CACHE" == 1 ]] || return 0
  feature_target="$root/features/$label"
  case "$feature_target" in "$CACHE_ROOT"/full/*/features/*) ;; *) echo "Refusing unexpected cleanup target: $feature_target" >&2; exit 1 ;; esac
  rm -rf -- "$feature_target"
}

candidate() {
  local id="${2:-}"
  [[ " $CANDIDATES " == *" $id "* ]] || { echo "Unknown candidate: $id" >&2; exit 2; }
  [[ -f "$SMOKE_MARKER" ]] || { echo "Missing passed smoke marker: $SMOKE_MARKER" >&2; exit 1; }
  require_clean_commit
  require_inputs
  lock_artifacts
  extract_one full "$id" "$SAMPLE_INDICES"
  train_candidate full "$id" 50 "$SAMPLE_INDICES"
  preserve_and_recycle "$id"
}

summarize() {
  run "$PYTHON" -u "$REPO_ROOT/scripts/probing/summarize_legacy_pre_sft_completion.py" \
    --results-root "$DURABLE_ROOT/results" --artifact-manifest "$ARTIFACT_MANIFEST" \
    --sample-indices "$SAMPLE_INDICES" --output-dir "$DURABLE_ROOT/summary"
}

case "$MODE" in
  preflight) preflight ;;
  smoke) smoke ;;
  candidate) candidate "$@" ;;
  summarize) summarize ;;
esac
