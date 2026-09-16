#!/usr/bin/env bash
# Complete local depth probes from the immutable GEOROPE features extracted on
# Snellius.  This wrapper never performs another VLM forward or extraction.
set -euo pipefail

MODE="${1:-}"
if [[ ! "$MODE" =~ ^(preflight|full)$ ]]; then
  echo "Usage: $0 {preflight|full}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-vlm3r}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/shaoruei/probe_cache/legacy_pre_sft_georope_resume_v1/full/GEOROPE}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/legacy_pre_sft_georope_local_resume_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/legacy_pre_sft_georope_local_resume_v1}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/home/shaoruei/probe_cache/legacy_pre_sft_georope_resume_v1/provenance/remote_artifact_manifest.json}"
MODEL_LABEL="c1_geo_rope_fusion"
SOURCE_COMMIT="de27d991985683dd9248c1a22f9e025e528560df"
EXPECTED_SAMPLE_SHA256="d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
EXPECTED_C1_SHA256="edb6ab3c255d0875e37cf6a18de078511fcfb61a37e3b102541e3f6219548f9c"
EXPECTED_GEOMETRY_C1_SHA256="6485ad95789948c95634ac0cb1cba03d18aa5cb901e9fc3ab4008fce42737254"
EXPECTED_FRAMES=2398
COVERAGE_AUDIT="$DURABLE_ROOT/provenance/resumed_feature_cache_audit.json"

source "$REPO_ROOT/scripts/probing/common_probe_layers.sh"
mkdir -p "$DURABLE_ROOT/provenance" "$LOG_ROOT"

run() {
  printf '[COMMAND] '; printf '%q ' "$@"; printf '\n'
  "$@"
}

require_source() {
  local level count provenance
  local -a levels
  provenance="$SOURCE_ROOT/features/$MODEL_LABEL/extraction_provenance.json"
  [[ -f "$provenance" ]] || { echo "Missing transferred extraction provenance: $provenance" >&2; exit 1; }
  [[ -f "$SAMPLE_INDICES" ]] || { echo "Missing fixed sample indices: $SAMPLE_INDICES" >&2; exit 1; }
  [[ -f "$SOURCE_MANIFEST" ]] || { echo "Missing transferred source artifact manifest: $SOURCE_MANIFEST" >&2; exit 1; }
  jq -e \
    --arg commit "$SOURCE_COMMIT" --arg split "$EXPECTED_SAMPLE_SHA256" \
    --arg c1 "$EXPECTED_C1_SHA256" --arg geometry_c1 "$EXPECTED_GEOMETRY_C1_SHA256" \
    '.git_commit == $commit and .git_worktree_dirty == false and
     .sample_indices_sha256 == $split and .no_vlm3r_sft_adapter_loaded == true and
     (.requested_feature_levels | length == 15) and
     .c1_calibration_sha256 == $c1 and .geometry_c1_calibration_sha256 == $geometry_c1' \
    "$provenance" >/dev/null
  IFS=',' read -r -a levels <<< "$PRE_SFT_FULL_FEATURE_LEVELS_CSV"
  for level in "${levels[@]}"; do
    count="$(find "$SOURCE_ROOT/features/$MODEL_LABEL/$level" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)"
    [[ "$count" -eq "$EXPECTED_FRAMES" ]] || {
      echo "Incomplete transferred feature level $level: $count/$EXPECTED_FRAMES" >&2
      exit 1
    }
  done
  for level in gt_depth metadata; do
    count="$(find "$SOURCE_ROOT/$level" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)"
    [[ "$count" -eq "$EXPECTED_FRAMES" ]] || {
      echo "Incomplete transferred $level cache: $count/$EXPECTED_FRAMES" >&2
      exit 1
    }
  done
}

audit_source_coverage() {
  if [[ -f "$COVERAGE_AUDIT" ]]; then
    jq -e '.status == "PASS_WITH_EXPLICIT_RESUME_PROVENANCE" and .fixed_videos == 1199 and .fixed_target_frames == 2398' \
      "$COVERAGE_AUDIT" >/dev/null
    return 0
  fi
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/audit_resumed_depth_feature_cache.py" \
    --cache-root "$SOURCE_ROOT" --model-label "$MODEL_LABEL" --sample-indices "$SAMPLE_INDICES" \
    --artifact-manifest "$SOURCE_MANIFEST" \
    --extraction-provenance "$SOURCE_ROOT/features/$MODEL_LABEL/extraction_provenance.json" \
    --candidate GEOROPE --source-job-ids 26749627,26761797 --output "$COVERAGE_AUDIT"
}

require_clean_trainer() {
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "Local GEOROPE resume requires a clean worktree" >&2
    exit 1
  }
  git -C "$REPO_ROOT" diff --quiet "$SOURCE_COMMIT"..HEAD -- scripts/probing/train_depth_probes.py || {
    echo "Probe trainer differs from the remote extraction commit" >&2
    exit 1
  }
}

validate_metrics() {
  local level metrics count
  local -a levels
  count=0
  IFS=',' read -r -a levels <<< "$PRE_SFT_FULL_FEATURE_LEVELS_CSV"
  for level in "${levels[@]}"; do
    metrics="$SOURCE_ROOT/probes/$MODEL_LABEL/$level/metrics.json"
    jq -e '.mae >= 0 and .mae < 1000000 and .absrel >= 0 and .delta125 >= 0 and .delta125 <= 1 and .num_tokens == 75656' \
      "$metrics" >/dev/null
    count=$((count + 1))
  done
  [[ "$count" -eq 15 ]]
}

train_level() {
  local level="$1" gpu="$2"
  env CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/legacy_georope_resume_mpl conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$SOURCE_ROOT" --sample-indices "$SAMPLE_INDICES" --probe-subdir probes \
    --model-labels "$MODEL_LABEL" --feature-levels "$level" --epochs 50 --batch-size 32 --lr 1e-3 \
    --early-stop-patience 10 --num-workers 0 --probe-seed 0 \
    --experiment-variant legacy_GEOROPE_remote_de27_local_probe_resume --device cuda:0 \
    --no-write-aggregate --skip-existing 2>&1 | tee "$LOG_ROOT/${level}_probe.log"
}

train_missing_levels() {
  local levels index first second first_pid second_pid
  IFS=',' read -r -a levels <<< "$PRE_SFT_FULL_FEATURE_LEVELS_CSV"
  index=0
  while [[ "$index" -lt "${#levels[@]}" ]]; do
    first="${levels[$index]}"; train_level "$first" 0 & first_pid=$!; index=$((index + 1))
    if [[ "$index" -lt "${#levels[@]}" ]]; then
      second="${levels[$index]}"; train_level "$second" 1 & second_pid=$!
      wait "$first_pid"; wait "$second_pid"; index=$((index + 1))
    else
      wait "$first_pid"
    fi
  done
}

preserve_results() {
  local destination="$DURABLE_ROOT/results/GEOROPE"
  [[ ! -e "$destination" ]] || { echo "Refusing to overwrite durable result: $destination" >&2; exit 1; }
  mkdir -p "$destination"
  cp -a "$SOURCE_ROOT/probes/$MODEL_LABEL" "$destination/probes"
  cp -a "$SOURCE_ROOT/features/$MODEL_LABEL/extraction_provenance.json" "$destination/extraction_provenance.json"
  cp -a "$SOURCE_MANIFEST" "$DURABLE_ROOT/provenance/remote_artifact_manifest.json"
  printf '%s\n' "$SOURCE_COMMIT" >"$DURABLE_ROOT/provenance/source_extraction_commit.txt"
  sha256sum "$SOURCE_ROOT/features/$MODEL_LABEL/extraction_provenance.json" >"$DURABLE_ROOT/provenance/source_extraction_provenance_sha256.txt"
}

preflight() {
  require_clean_trainer
  require_source
  audit_source_coverage
  echo "[PASS] GEOROPE remote-feature local-probe resume preflight"
}

full() {
  preflight
  train_missing_levels
  validate_metrics
  preserve_results
  echo "[PASS] GEOROPE remote-feature local-probe resume"
}

case "$MODE" in
  preflight) preflight ;;
  full) full ;;
esac
