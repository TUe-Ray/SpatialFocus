#!/usr/bin/env bash
# Shared configuration for the existing A/F/G reference architectures.
set -euo pipefail

ARCH_ID="${REFERENCE_FUSION_ID:?Set REFERENCE_FUSION_ID to A, F, or G}"

export MODEL_LORA_ENABLE=True
export MODEL_LORA_R=128
export MODEL_LORA_ALPHA=256
export MODEL_SPATIAL_TOWER=cut3r
export MODEL_SPATIAL_TOWER_SELECT_FEATURE=all_tokens
export MODEL_SPATIAL_FEATURE_DIM=768
export MODEL_CUT3R_SPATIALSTACK_PROJECTOR_TYPE=token_mlp
export MODEL_CUT3R_SPATIALSTACK_PROJECTOR_BINDING=source_specific
export MODEL_USE_CUT3R_CAMERA_TOKENS=False
export MODEL_USE_POINTMAP_SUPERVISION=False
export MODEL_USE_AUXILIARY_GEOMETRY_HEAD=False
export MODEL_USE_AUXILIARY_GEOMETRY_LOSS=False
export MODEL_USE_BEV_SUPERVISION=False
export MODEL_USE_DEPTH_SUPERVISION=False
export MODEL_LLM_VISUAL_3D_ROPE_ENABLE=False
export CHECKPOINT_MILESTONE_RATIOS="${CHECKPOINT_MILESTONE_RATIOS:-0.01,0.05,0.25,0.50}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"

case "$ARCH_ID" in
    A)
        export NOTE="Reference A: legacy VLM3R CUT3R cross-attention fusion before the LLM."
        export SUFFIX="reference_A_vlm3r_prellm_cross_attn"
        export MODEL_SPATIAL_TOWER_PREEXTRACTED_ONLY=False
        export MODEL_USE_CUT3R_SPATIALSTACK=False
        export MODEL_TUNE_CUT3R_SPATIALSTACK=False
        export MODEL_CUT3R_SPATIALSTACK_LAYERS=12
        export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS=0
        export MODEL_FUSION_BLOCK=cross_attention
        export MODEL_TUNE_FUSION_BLOCK=True
        export MODEL_TUNE_MM_MLP_ADAPTER=True
        ;;
    F)
        export NOTE="Reference F: SpatialStack Add dec6/9/12 before LLM layers 0/1/2."
        export SUFFIX="reference_F_spatialstack_add_dec6_9_12_llm0_1_2"
        export MODEL_SPATIAL_TOWER_PREEXTRACTED_ONLY=True
        export MODEL_USE_CUT3R_SPATIALSTACK=True
        export MODEL_TUNE_CUT3R_SPATIALSTACK=True
        export MODEL_CUT3R_SPATIALSTACK_LAYERS=6,9,12
        export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS=0,1,2
        export MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE=add
        export MODEL_FUSION_BLOCK=""
        export MODEL_TUNE_FUSION_BLOCK=False
        export MODEL_TUNE_MM_MLP_ADAPTER=False
        ;;
    G)
        export NOTE="Reference G: SpatialStack Cross-Attn dec6/9/12 before LLM layers 0/1/2."
        export SUFFIX="reference_G_spatialstack_cross_attn_dec6_9_12_llm0_1_2"
        export MODEL_SPATIAL_TOWER_PREEXTRACTED_ONLY=True
        export MODEL_USE_CUT3R_SPATIALSTACK=True
        export MODEL_TUNE_CUT3R_SPATIALSTACK=True
        export MODEL_CUT3R_SPATIALSTACK_LAYERS=6,9,12
        export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS=0,1,2
        export MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE=cross_attn
        export MODEL_FUSION_BLOCK=""
        export MODEL_TUNE_FUSION_BLOCK=False
        export MODEL_TUNE_MM_MLP_ADAPTER=False
        ;;
    *)
        echo "[ERROR] Unsupported REFERENCE_FUSION_ID=$ARCH_ID (expected A/F/G)." >&2
        exit 2
        ;;
esac

case "${REFERENCE_FUSION_CONFIG_ONLY:-False}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
        return 0 2>/dev/null || exit 0
        ;;
esac

export TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-${SUFFIX}_${SLURM_JOB_ID:-manual}}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "$SCRIPT_DIR/train_cut3r_spatialstack.sh"
