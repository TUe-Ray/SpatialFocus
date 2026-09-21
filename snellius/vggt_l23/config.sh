#!/usr/bin/env bash

readonly REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
readonly VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
readonly SOURCE_ROOT="/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt"
readonly STAGING_ROOT="/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_staging"
readonly FINAL_ROOT="/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23"
readonly ARTIFACTS_ROOT="/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_artifacts"
readonly LOG_ROOT="$ARTIFACTS_ROOT/logs"
readonly CONVERTER="$REPO_DIR/scripts/extraction/convert_vggt_l23.py"
