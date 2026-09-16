#!/usr/bin/env bash
# Wait for the audited GEOROPE transfer, finish its local probes, and merge all
# seven legacy pre-SFT candidates into one provenance-aware result bundle.
set -euo pipefail

MODE="${1:-}"
if [[ ! "$MODE" =~ ^(status|finalize)$ ]]; then
  echo "Usage: $0 {status|finalize}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-vlm3r}"
TRANSFER_UNIT="${TRANSFER_UNIT:-spatialfocus-georope-cache-transfer.service}"
REMOTE_LOGIN="${REMOTE_LOGIN:-shuang@snellius.surf.nl}"
REMOTE_MANIFEST="${REMOTE_MANIFEST:-/home/shuang/proxy_outputs/legacy_pre_sft_completion_a100_v1_retry1/provenance/artifact_manifest.json}"
TRANSFER_ROOT="${TRANSFER_ROOT:-/home/shaoruei/probe_cache/legacy_pre_sft_georope_resume_v1}"
SOURCE_MANIFEST="$TRANSFER_ROOT/provenance/remote_artifact_manifest.json"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
GEO_DURABLE_ROOT="${GEO_DURABLE_ROOT:-/home/shaoruei/probe_outputs/legacy_pre_sft_georope_local_resume_v1}"
MERGED_ROOT="${MERGED_ROOT:-/home/shaoruei/probe_outputs/legacy_pre_sft_completion_merged_v1}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-/home/shaoruei/probe_outputs/legacy_pre_sft_completion_v1}"
LOCAL_REST_ROOT="${LOCAL_REST_ROOT:-/home/shaoruei/probe_outputs/legacy_pre_sft_completion_local_rest_v1}"

run() {
  printf '[COMMAND] '; printf '%q ' "$@"; printf '\n'
  "$@"
}

transfer_status() {
  systemctl --user show "$TRANSFER_UNIT" \
    --property=ActiveState,SubState,Result,ExecMainStatus,MainPID 2>/dev/null || true
  find "$TRANSFER_ROOT/full/GEOROPE/features/c1_geo_rope_fusion" -name '*.pt' -type f 2>/dev/null | wc -l
  du -sh "$TRANSFER_ROOT/full/GEOROPE" 2>/dev/null || true
}

wait_for_transfer() {
  while systemctl --user is-active --quiet "$TRANSFER_UNIT"; do
    transfer_status
    sleep 30
  done
  transfer_status
}

retrieve_source_manifest() {
  local temporary
  mkdir -p "$(dirname "$SOURCE_MANIFEST")"
  temporary="$(mktemp /tmp/georope_remote_manifest.XXXXXX.json)"
  run scp -o BatchMode=yes "$REMOTE_LOGIN:$REMOTE_MANIFEST" "$temporary"
  if [[ -f "$SOURCE_MANIFEST" ]]; then
    cmp --silent "$temporary" "$SOURCE_MANIFEST" || {
      echo "Existing remote artifact manifest differs: $SOURCE_MANIFEST" >&2
      rm -f -- "$temporary"
      exit 1
    }
  else
    cp "$temporary" "$SOURCE_MANIFEST"
  fi
  rm -f -- "$temporary"
}

summarize_all() {
  run conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_legacy_pre_sft_multisource.py" \
    --sample-indices "$SAMPLE_INDICES" \
    --source SS012 "$ORIGINAL_ROOT/provenance/artifact_manifest.json" "$ORIGINAL_ROOT/results/SS012" - \
    --source SS123 "$ORIGINAL_ROOT/provenance/artifact_manifest.json" "$ORIGINAL_ROOT/results/SS123" - \
    --source SS036 "$ORIGINAL_ROOT/provenance/artifact_manifest.json" "$ORIGINAL_ROOT/results/SS036" - \
    --source SSCROSS "$LOCAL_REST_ROOT/provenance/artifact_manifest.json" "$LOCAL_REST_ROOT/results/SSCROSS" - \
    --source VLM3R "$LOCAL_REST_ROOT/provenance/artifact_manifest.json" "$LOCAL_REST_ROOT/results/VLM3R" - \
    --source GEOROPE "$GEO_DURABLE_ROOT/provenance/remote_artifact_manifest.json" "$GEO_DURABLE_ROOT/results/GEOROPE" "$GEO_DURABLE_ROOT/provenance/resumed_feature_cache_audit.json" \
    --source SELECTIVE "$LOCAL_REST_ROOT/provenance/artifact_manifest.json" "$LOCAL_REST_ROOT/results/SELECTIVE" - \
    --output-dir "$MERGED_ROOT/summary"
  jq -e '.post_sft_state_loaded == false and (.sources | length) == 7 and (.rows | length) == 105' \
    "$MERGED_ROOT/summary/results.json" >/dev/null
}

finalize() {
  wait_for_transfer
  retrieve_source_manifest
  run bash "$REPO_ROOT/scripts/probing/run_legacy_pre_sft_georope_local_resume.sh" full
  summarize_all
  echo "[PASS] seven-model legacy pre-SFT completion finalized: $MERGED_ROOT/summary/results.json"
}

case "$MODE" in
  status) transfer_status ;;
  finalize) finalize ;;
esac
