#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

export PYTHONNOUSERSITE=1
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"

python "$VALIDATOR" \
  --spec training_repair "$TRAIN_REPAIR_IDS" "$RAW_TRAIN_ROOT/scannetpp/videos" "$SCANNETPP_OUTPUT" \
  --spec eval_scannet "$EVAL_SCANNET_IDS" "$RAW_VSI_ROOT/scannet" "$SCANNET_OUTPUT" \
  --spec eval_scannetpp "$EVAL_SCANNETPP_IDS" "$RAW_VSI_ROOT/scannetpp" "$SCANNETPP_OUTPUT" \
  --spec eval_arkitscenes "$EVAL_ARKITSCENES_IDS" "$RAW_VSI_ROOT/arkitscenes" "$ARKITSCENES_OUTPUT" \
  --report "$REPORT_ROOT/staging_all_412.json"
