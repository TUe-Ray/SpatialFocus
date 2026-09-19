#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

[[ "$(hostname -f)" == *.snellius.surf.nl ]] || {
  echo "This preparation script is Snellius-only." >&2
  exit 2
}

mkdir -p \
  "$VIEW_ROOT" "$LOG_ROOT" "$REPORT_ROOT" "$PROVENANCE_ROOT" \
  "$STAGING_ROOT/_runtime_cache/matplotlib" \
  "$SCANNET_OUTPUT" "$SCANNETPP_OUTPUT" "$ARKITSCENES_OUTPUT"

sha256sum -c "$HANDOFF_ROOT/checksums.sha256"
[[ "$(git -C "$REPO_DIR/third_party/VGGT" rev-parse HEAD)" == "$VGGT_SOURCE_COMMIT" ]]
[[ "$(sha256sum "$VGGT_WEIGHTS/model.safetensors" | awk '{print $1}')" == "$VGGT_WEIGHTS_SHA256" ]]

python "$HANDOFF_ROOT/make_input_view.py" \
  --ids "$TRAIN_REPAIR_IDS" \
  --raw-root "$RAW_TRAIN_ROOT/scannetpp/videos" \
  --view-dir "$TRAIN_REPAIR_VIEW"
python "$HANDOFF_ROOT/make_input_view.py" \
  --ids "$EVAL_SCANNET_IDS" \
  --raw-root "$RAW_VSI_ROOT/scannet" \
  --view-dir "$EVAL_SCANNET_VIEW"
python "$HANDOFF_ROOT/make_input_view.py" \
  --ids "$EVAL_SCANNETPP_IDS" \
  --raw-root "$RAW_VSI_ROOT/scannetpp" \
  --view-dir "$EVAL_SCANNETPP_VIEW"
python "$HANDOFF_ROOT/make_input_view.py" \
  --ids "$EVAL_ARKITSCENES_IDS" \
  --raw-root "$RAW_VSI_ROOT/arkitscenes" \
  --view-dir "$EVAL_ARKITSCENES_VIEW"

provenance_file="$PROVENANCE_ROOT/preparation.txt"
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname -f)"
  echo "repository_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "vggt_source_commit=$(git -C "$REPO_DIR/third_party/VGGT" rev-parse HEAD)"
  echo "vggt_checkpoint=$VGGT_WEIGHTS/model.safetensors"
  echo "vggt_checkpoint_sha256=$(sha256sum "$VGGT_WEIGHTS/model.safetensors" | awk '{print $1}')"
  echo "processor_config=$PROCESSOR_CONFIG"
  echo "processor_config_sha256=$(sha256sum "$PROCESSOR_CONFIG" | awk '{print $1}')"
  echo "raw_train_root=$RAW_TRAIN_ROOT"
  echo "raw_vsi_root=$RAW_VSI_ROOT"
  echo "staging_root=$STAGING_ROOT"
  echo "final_root=$FINAL_ROOT"
  echo "training_repair_count=$(wc -l < "$TRAIN_REPAIR_IDS")"
  echo "eval_scannet_count=$(wc -l < "$EVAL_SCANNET_IDS")"
  echo "eval_scannetpp_count=$(wc -l < "$EVAL_SCANNETPP_IDS")"
  echo "eval_arkitscenes_count=$(wc -l < "$EVAL_ARKITSCENES_IDS")"
  echo "smoke_command=sbatch $SCRIPT_DIR/smoke.sbatch"
  echo "full_scannet_command=sbatch $SCRIPT_DIR/extract_scannet.sbatch"
  echo "full_scannetpp_command=sbatch $SCRIPT_DIR/extract_scannetpp.sbatch"
  echo "full_arkitscenes_command=sbatch $SCRIPT_DIR/extract_arkitscenes.sbatch"
} > "$provenance_file"

echo "Prepared exact input views and provenance under $STAGING_ROOT"
