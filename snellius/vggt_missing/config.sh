#!/usr/bin/env bash
# Fixed Snellius paths for the Leonardo-compatible VGGT repair extraction.

readonly REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
readonly VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"

readonly HF_HOME="/home/geusdd/.cache/huggingface"
readonly HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
readonly VGGT_REVISION="860abec7937da0a4c03c41d3c269c366e82abdf9"
readonly VGGT_WEIGHTS="$HUGGINGFACE_HUB_CACHE/models--facebook--VGGT-1B/snapshots/$VGGT_REVISION"
readonly VGGT_WEIGHTS_SHA256="f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
readonly VGGT_SOURCE_COMMIT="44b3afbd1869d8bde4894dd8ea1e293112dd5eba"

readonly RAW_TRAIN_ROOT="/scratch-shared/geusdd/VLM3R/data/vlm3r"
readonly RAW_VSI_ROOT="/scratch-shared/geusdd/VLM3R/hf_cache/vsibench"
readonly FINAL_ROOT="/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt"
readonly STAGING_ROOT="/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_missing_staging"
readonly VIEW_ROOT="$STAGING_ROOT/_input_views"
readonly LOG_ROOT="$STAGING_ROOT/_logs"
readonly REPORT_ROOT="$STAGING_ROOT/_reports"
readonly PROVENANCE_ROOT="$STAGING_ROOT/_provenance"

readonly EXTRACTOR="$REPO_DIR/scripts/extraction/extract_vggt_features.py"
readonly VALIDATOR="$REPO_DIR/scripts/extraction/validate_vggt_sidecars.py"
readonly PROCESSOR_CONFIG="$REPO_DIR/vggt_extraction_handoff/processor_config.json"
readonly HANDOFF_ROOT="$REPO_DIR/vggt_extraction_handoff"

readonly TRAIN_REPAIR_IDS="$HANDOFF_ROOT/training_repair_ids.txt"
readonly EVAL_SCANNET_IDS="$HANDOFF_ROOT/eval_scannet_ids.txt"
readonly EVAL_SCANNETPP_IDS="$HANDOFF_ROOT/eval_scannetpp_ids.txt"
readonly EVAL_ARKITSCENES_IDS="$HANDOFF_ROOT/eval_arkitscenes_ids.txt"

readonly TRAIN_REPAIR_VIEW="$VIEW_ROOT/train_repair_scannetpp"
readonly EVAL_SCANNET_VIEW="$VIEW_ROOT/eval_scannet"
readonly EVAL_SCANNETPP_VIEW="$VIEW_ROOT/eval_scannetpp"
readonly EVAL_ARKITSCENES_VIEW="$VIEW_ROOT/eval_arkitscenes"

readonly SCANNET_OUTPUT="$STAGING_ROOT/scannet"
readonly SCANNETPP_OUTPUT="$STAGING_ROOT/scannetpp"
readonly ARKITSCENES_OUTPUT="$STAGING_ROOT/arkitscenes"
