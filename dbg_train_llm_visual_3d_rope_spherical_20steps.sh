#!/bin/bash
#SBATCH --job-name=DBG_llm_visual_3d_rope_20step
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclusive

set -euo pipefail

export NOTE="${NOTE:-DBG 20-step speed probe for LLM visual-token-only 3D RoPE; CUT3R reference geometry.}"

# Match the official single-shot run configuration.
export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_patch_only}"
export MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
export MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"

export MODEL_LLM_VISUAL_3D_ROPE_ENABLE="${MODEL_LLM_VISUAL_3D_ROPE_ENABLE:-True}"
export MODEL_LLM_VISUAL_3D_ROPE_ALPHA="${MODEL_LLM_VISUAL_3D_ROPE_ALPHA:-1.0}"
export MODEL_LLM_VISUAL_3D_ROPE_MODE="${MODEL_LLM_VISUAL_3D_ROPE_MODE:-spherical}"
export MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT="${MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT:-2,1,2}"
export MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH="${MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH:-10.0}"
export MODEL_LLM_VISUAL_3D_ROPE_LAYERS="${MODEL_LLM_VISUAL_3D_ROPE_LAYERS:-all}"
export MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE="${MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE:-point_maps_ref}"
export MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE="${MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE:-False}"
export MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS="${MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS:-True}"
export MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS="${MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS:-first_middle_last}"
export MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION="${MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION:-True}"

# Speed probe controls: normal global batch and 5-step reporting, capped at 20
# optimizer steps.
export MAX_STEPS="${MAX_STEPS:-20}"
export TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-128}"
export TRAIN_DATA_PERCENTAGE="${TRAIN_DATA_PERCENTAGE:-100}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export REPORT_TO="${REPORT_TO:-none}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-6}"

export MODEL_TORCH_COMPILE="${MODEL_TORCH_COMPILE:-False}"
export MODEL_GRADIENT_CHECKPOINTING="${MODEL_GRADIENT_CHECKPOINTING:-True}"

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
exec bash "$REPO_DIR/train_geo_rope_fusion_cut3r.sh"
