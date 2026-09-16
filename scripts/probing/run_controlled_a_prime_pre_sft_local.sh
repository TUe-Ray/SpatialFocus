#!/usr/bin/env bash
# Dedicated non-C1 A-prime full-policy depth probe on mps-edu-06.
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
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
DATA_YAML="${DATA_YAML:-$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/controlled_a_prime_pre_sft_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/controlled_a_prime_pre_sft_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/controlled_a_prime_pre_sft_v1}"
MANIFEST="$DURABLE_ROOT/provenance/experiment_manifest.json"
SMOKE_MANIFEST="$DURABLE_ROOT/provenance/smoke_1train_1val.json"
SMOKE_MARKER="$DURABLE_ROOT/provenance/smoke_verification.json"
GPU_WEIGHT_BUDGET="${PRE_SFT_GPU_WEIGHT_BUDGET:-4GiB}"
CPU_OFFLOAD_BUDGET="${PRE_SFT_CPU_OFFLOAD_BUDGET:-45GiB}"
RECYCLE_FEATURE_CACHE="${RECYCLE_FEATURE_CACHE:-1}"
OFFICIAL_SPLIT_SHA256="d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"

source "$REPO_ROOT/scripts/probing/common_probe_layers.sh"
mkdir -p "$CACHE_ROOT" "$DURABLE_ROOT/provenance" "$LOG_ROOT"

run() {
  printf '[COMMAND] '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

require_inputs() {
  local path
  for path in "$BASE_MODEL/config.json" "$SIGLIP_MODEL/config.json" "$SAMPLE_INDICES" "$DATA_YAML" \
    "$FEATURE_ROOT/scannet/spatial_features/scene0384_00.pt"; do
    [[ -e "$path" ]] || { echo "Missing required A-prime input: $path" >&2; exit 1; }
  done
  [[ -d "$FORWARD_ROOT" && -d "$TARGET_ROOT" && -d "$FEATURE_ROOT" ]] || {
    echo "Missing frame, target, or CUT3R root" >&2
    exit 1
  }
  local actual_split
  actual_split="$(sha256sum "$SAMPLE_INDICES" | cut -d' ' -f1)"
  [[ "$actual_split" == "$OFFICIAL_SPLIT_SHA256" ]] || {
    echo "Wrong formal ScanNet split: $actual_split" >&2
    exit 1
  }
  local forbidden
  forbidden="$(find "$BASE_MODEL" -type f \( -name adapter_model.bin -o -name non_lora_trainables.bin -o -name adapter_config.json \) -print -quit)"
  [[ -z "$forbidden" ]] || { echo "Base model contains forbidden post-SFT state: $forbidden" >&2; exit 1; }
}

require_gpu() {
  local purpose="$1"
  nvidia-smi --id="$GPU" --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
  run env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
    --physical-gpu-id "$GPU" --output "$DURABLE_ROOT/provenance/gpu_${GPU}_${purpose}_readiness.json"
}

prepare_manifest() {
  local -a check_args=()
  [[ -f "$MANIFEST" ]] && check_args=(--check-existing)
  run conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/prepare_controlled_a_prime_pre_sft.py" \
    --base-model "$BASE_MODEL" --siglip-model "$SIGLIP_MODEL" --feature-root "$FEATURE_ROOT" \
    --sample-indices "$SAMPLE_INDICES" --output "$MANIFEST" "${check_args[@]}"
}

preflight() {
  require_inputs
  run conda run -n "$ENV_NAME" python -m py_compile \
    "$REPO_ROOT/llava/model/controlled_fusion_pre_sft.py" \
    "$REPO_ROOT/scripts/diagnose_layerwise_spatial_hidden_scan.py" \
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    "$REPO_ROOT/scripts/probing/prepare_controlled_a_prime_pre_sft.py" \
    "$REPO_ROOT/scripts/probing/verify_controlled_a_prime_pre_sft_smoke.py" \
    "$REPO_ROOT/scripts/probing/summarize_controlled_a_prime_pre_sft.py"
  prepare_manifest
  require_gpu preflight
  echo "[PASS] controlled A-prime pre-SFT preflight"
}

make_smoke_manifest() {
  [[ -f "$SMOKE_MANIFEST" ]] && return 0
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
    --sample-indices "$SAMPLE_INDICES" --output "$SMOKE_MANIFEST" --train-videos 1 --val-videos 1
}

extract_one() {
  local namespace="$1" id="$2" samples="$3" label mode output log
  if [[ "$id" == BASE ]]; then
    label="pre_sft_base_vlm"
    mode="pre_sft_base_vlm"
  else
    label="controlled_a_prime"
    mode="pre_sft_fusion"
  fi
  output="$CACHE_ROOT/$namespace/$id"
  log="$LOG_ROOT/${namespace}_${id}_extract.log"
  mkdir -p "$output"
  local -a architecture_args=()
  if [[ "$id" == A_prime ]]; then
    architecture_args=(
      --pre-sft-fusion-variant controlled_a_prime --fusion-init-seed 42
      --feature-root "$FEATURE_ROOT" --spatial-features-subdir '12:spatial_features'
      --spatialstack-cut3r-layers 12 --spatialstack-llm-layers ''
    )
  fi
  echo "[RUN] A-prime pre-SFT extraction namespace=$namespace candidate=$id GPUs=$CUDA_DEVICES output=$output log=$log"
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" MPLCONFIGDIR=/tmp/a_prime_presft_mpl \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
    --model-label "$label" --model-loading-mode "$mode" "${architecture_args[@]}" \
    --model-path "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" \
    --feature-levels "$PRE_SFT_FULL_FEATURE_LEVELS_CSV" --sample-indices "$samples" --output-root "$output" \
    --train-data-json "$DATA_YAML" --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
    --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" --frames-upbound 32 \
    --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
    --runtime-root "$output/runtime/$label" --pre-sft-gpu-weight-budget "$GPU_WEIGHT_BUDGET" \
    --pre-sft-cpu-offload-budget "$CPU_OFFLOAD_BUDGET" --assert-first-video --resume 2>&1 | tee "$log"
}

train_level() {
  local namespace="$1" id="$2" level="$3" physical_gpu="$4" epochs="$5" label root log samples
  [[ "$id" == BASE ]] && label="pre_sft_base_vlm" || label="controlled_a_prime"
  root="$CACHE_ROOT/$namespace/$id"
  log="$LOG_ROOT/${namespace}_${id}_${level}_probe.log"
  [[ "$namespace" == smoke ]] && samples="$SMOKE_MANIFEST" || samples="$SAMPLE_INDICES"
  env CUDA_VISIBLE_DEVICES="$physical_gpu" MPLCONFIGDIR=/tmp/a_prime_presft_mpl \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$root" --sample-indices "$samples" --probe-subdir probes \
    --model-labels "$label" --feature-levels "$level" --epochs "$epochs" --batch-size 32 \
    --lr 1e-3 --early-stop-patience 10 --num-workers 0 --probe-seed 0 \
    --experiment-variant "controlled_a_prime_pre_sft" --device cuda:0 \
    --no-write-aggregate --skip-existing 2>&1 | tee "$log"
}

train_all_levels() {
  local namespace="$1" id="$2" epochs="$3" index first second first_pid second_pid
  local -a levels
  IFS=',' read -r -a levels <<< "$PRE_SFT_FULL_FEATURE_LEVELS_CSV"
  index=0
  while [[ "$index" -lt "${#levels[@]}" ]]; do
    first="${levels[$index]}"
    train_level "$namespace" "$id" "$first" 0 "$epochs" & first_pid=$!
    index=$((index + 1))
    if [[ "$index" -lt "${#levels[@]}" ]]; then
      second="${levels[$index]}"
      train_level "$namespace" "$id" "$second" 1 "$epochs" & second_pid=$!
      wait "$first_pid"
      wait "$second_pid"
      index=$((index + 1))
    else
      wait "$first_pid"
    fi
  done
}

smoke() {
  require_inputs
  prepare_manifest
  require_gpu smoke
  make_smoke_manifest
  extract_one smoke BASE "$SMOKE_MANIFEST"
  train_all_levels smoke BASE 2
  extract_one smoke A_prime "$SMOKE_MANIFEST"
  train_all_levels smoke A_prime 2
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_controlled_a_prime_pre_sft_smoke.py" \
    --cache-root "$CACHE_ROOT/smoke" --manifest "$MANIFEST" \
    --sample-indices "$SMOKE_MANIFEST" --output "$SMOKE_MARKER"
}

preserve_one() {
  local id="$1" label root destination
  [[ "$id" == BASE ]] && label="pre_sft_base_vlm" || label="controlled_a_prime"
  root="$CACHE_ROOT/full/$id"
  destination="$DURABLE_ROOT/results/$id"
  [[ ! -e "$destination" ]] || { echo "Refusing to overwrite durable A-prime result: $destination" >&2; exit 1; }
  mkdir -p "$destination"
  cp -a "$root/probes/$label" "$destination/probes"
  cp -a "$root/features/$label/extraction_provenance.json" "$destination/extraction_provenance.json"
}

recycle_one() {
  local id="$1" label target
  [[ "$RECYCLE_FEATURE_CACHE" == 1 ]] || return 0
  [[ "$id" == BASE ]] && label="pre_sft_base_vlm" || label="controlled_a_prime"
  target="$CACHE_ROOT/full/$id/features/$label"
  case "$target" in
    "$CACHE_ROOT/full/BASE/features/pre_sft_base_vlm"|"$CACHE_ROOT/full/A_prime/features/controlled_a_prime") ;;
    *) echo "Refusing unexpected A-prime feature cleanup target: $target" >&2; exit 1 ;;
  esac
  [[ -d "$target" ]] || return 0
  echo "[RECYCLE] removing regeneratable feature tensors after durable copy: $target"
  rm -rf -- "$target"
}

summarize() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_controlled_a_prime_pre_sft.py" \
    --results-root "$DURABLE_ROOT/results" --manifest "$MANIFEST" \
    --sample-indices "$SAMPLE_INDICES" --output-dir "$DURABLE_ROOT/summary"
}

full() {
  require_inputs
  [[ -f "$SMOKE_MARKER" ]] || { echo "Run '$0 smoke' successfully before the formal sweep." >&2; exit 1; }
  prepare_manifest
  require_gpu full
  extract_one full BASE "$SAMPLE_INDICES"
  train_all_levels full BASE 50
  preserve_one BASE
  recycle_one BASE
  extract_one full A_prime "$SAMPLE_INDICES"
  train_all_levels full A_prime 50
  preserve_one A_prime
  recycle_one A_prime
  summarize
}

case "$MODE" in
  preflight) preflight ;;
  smoke) smoke ;;
  full) full ;;
  summarize) summarize ;;
esac
