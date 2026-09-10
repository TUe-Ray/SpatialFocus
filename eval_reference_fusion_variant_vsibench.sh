#!/usr/bin/env bash
# Shared VSI-Bench handoff for the existing F/G SpatialStack references.
set -euo pipefail

ARCH_ID="${REFERENCE_FUSION_ID:?Set REFERENCE_FUSION_ID to F or G}"
export PRESERVE_CHECKPOINT_CONFIG=True
export EXPECTED_USE_CUT3R_SPATIALSTACK=True
export EXPECTED_FUSION_BLOCK=none
export EXPECTED_PRE_PROJECTOR_ADD_SOURCE_LAYER=""
export EXPECTED_CUT3R_SPATIALSTACK_LAYERS=6,9,12
export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS=0,1,2
export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_TYPE=token_mlp
export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_BINDING=source_specific

case "$ARCH_ID" in
    F) export EXPECTED_CUT3R_SPATIALSTACK_FUSION_TYPE=add ;;
    G) export EXPECTED_CUT3R_SPATIALSTACK_FUSION_TYPE=cross_attn ;;
    *) echo "[ERROR] Unsupported REFERENCE_FUSION_ID=$ARCH_ID (expected F/G)." >&2; exit 2 ;;
esac

export CUT3R_SPATIALSTACK_LAYERS=6,9,12
export CUT3R_SPATIALSTACK_LLM_LAYERS=0,1,2
REPO_ROOT="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
exec bash "$REPO_ROOT/eval_spatialstack_vsibench.sh"
