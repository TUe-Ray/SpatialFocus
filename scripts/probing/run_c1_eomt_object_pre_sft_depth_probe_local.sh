#!/usr/bin/env bash
# C1 VLM3R full-K/V plus deterministic appended EoMT object tokens.
#
# This is a legacy partial-layer diagnostic, not a formal full-policy pre-SFT
# representation cache: it deliberately requests only the listed LLM and
# pre-LLM levels and therefore opts into --allow-incomplete-pre-sft-features.
set -euo pipefail
[[ "${1:-}" == "full" ]] || { echo "Usage: $0 full" >&2; exit 2; }
R="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"; C="${CACHE_ROOT:-/home/shaoruei/probe_cache/c1_eomt_object_pre_sft_v1}"; L="${LOG_ROOT:-$R/logs/c1_eomt_object_pre_sft_v1}"
B="/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2"; S="/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384"; F="/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1"; T="/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1"; X="/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features"; I="/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"; A="/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json"; E="/home/shaoruei/probe_cache/eomt_consumer_grid_v2"
mkdir -p "$C/full" "$L"
env CUDA_VISIBLE_DEVICES=0,1 conda run -n vlm3r python -u "$R/scripts/probing/extract_depth_probe_features.py" --model-loading-mode pre_sft_fusion --pre-sft-fusion-variant c1_eomt_object --c1-calibration-json "$A" --model-label c1_eomt_object --model-path "$B" --siglip-path "$S" --feature-preset llm_only --layers 0 1 2 3 6 9 15 21 27 --pre-llm-features fusion_output,projected_features --allow-incomplete-pre-sft-features --output-root "$C/full" --sample-indices "$I" --train-data-json "$R/scripts/probing/scannet_depth_probe_local_data.yaml" --feature-root "$X" --spatial-features-subdir spatial_features --forward-frames-root "$F" --probe-targets-root "$T" --image-folder "$F" --video-folder "$F" --eomt-consumer-cache-root "$E" --eomt-cache-validation "$E/validation.json" --frames-upbound 32 --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 --runtime-root "$C/full/runtime" --pre-sft-gpu-weight-budget 4GiB --pre-sft-cpu-offload-budget 45GiB --resume 2>&1 | tee "$L/full_extraction.log"
layers=(0 1 2 3 6 9 15 21 27)
for ((i=0; i<${#layers[@]}; i+=2)); do
  for slot in 0 1; do
    j=$((i+slot)); [[ $j -lt ${#layers[@]} ]] || continue; layer="${layers[$j]}"
    env CUDA_VISIBLE_DEVICES="$slot" conda run -n vlm3r python -u "$R/scripts/probing/train_depth_probes.py" --output-root "$C/full" --sample-indices "$I" --probe-subdir probes --model-labels c1_eomt_object --feature-levels "layer_$layer" --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0 --probe-seed 0 --experiment-variant c1_eomt_object --device cuda:0 --no-write-aggregate --skip-existing 2>&1 | tee "$L/probe_layer_$layer.log" &
  done
  wait
done
