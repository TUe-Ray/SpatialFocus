#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/shaoruei/SpatialFocus
OUTPUT_ROOT=/mnt/DATA_SSD/shaoruei/probing_data/internvl3_presft_probe_v2
SAMPLES=/home/shaoruei/probe_outputs/scannet_depth_layers_v1/full/provenance/scannet_sample_indices.json
PYTHON=/home/shaoruei/miniconda3/envs/vlm3r/bin/python
MANIFEST="$OUTPUT_ROOT/internvl3_8b_base_presft_run_manifest.json"
LOGME_LOG="$OUTPUT_ROOT/logme_common7.log"
LEVELS=visual_output,fusion_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27
LEVELS_GPU0=visual_output,projected_features,layer_0,layer_2,layer_6,layer_12,layer_18,layer_24
LEVELS_GPU1=fusion_output,layer_1,layer_3,layer_9,layer_15,layer_21,layer_27

cd "$REPO_ROOT"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing extraction manifest: $MANIFEST" >&2
  exit 1
fi
EXTRACT_PID=$(
  "$PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["pid"]))' "$MANIFEST"
)
while ps -p "$EXTRACT_PID" -o args= 2>/dev/null | rg -q 'extract_internvl3_presft_features\.py'; do
  sleep 30
done
"$PYTHON" -c '
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest.get("status") != "complete" or manifest.get("complete_videos") != 1199:
    raise SystemExit("Full 1,199-scene extraction did not complete; refusing partial scoring")
' "$MANIFEST"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u scripts/probing/run_internvl3_base_common7_logme.py \
  --output-root "$OUTPUT_ROOT" --sample-indices "$SAMPLES" \
  --device cuda:0 --block-frames 16 > "$LOGME_LOG" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u scripts/probing/train_depth_probes.py \
  --output-root "$OUTPUT_ROOT" --sample-indices "$SAMPLES" \
  --model-labels internvl3_8b_base_presft --feature-levels "$LEVELS_GPU0" \
  --probe-subdir probes --result-stem internvl3_8b_base_depth \
  --epochs 50 --batch-size 32 --lr 0.001 --early-stop-patience 10 \
  --probe-seed 0 --device cuda:0 --skip-existing --no-write-aggregate \
  > "$OUTPUT_ROOT/depth_probes_gpu0.log" 2>&1 &
PID_GPU0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u scripts/probing/train_depth_probes.py \
  --output-root "$OUTPUT_ROOT" --sample-indices "$SAMPLES" \
  --model-labels internvl3_8b_base_presft --feature-levels "$LEVELS_GPU1" \
  --probe-subdir probes --result-stem internvl3_8b_base_depth \
  --epochs 50 --batch-size 32 --lr 0.001 --early-stop-patience 10 \
  --probe-seed 0 --device cuda:0 --skip-existing --no-write-aggregate \
  > "$OUTPUT_ROOT/depth_probes_gpu1.log" 2>&1 &
PID_GPU1=$!
STATUS_GPU0=0
STATUS_GPU1=0
wait "$PID_GPU0" || STATUS_GPU0=$?
wait "$PID_GPU1" || STATUS_GPU1=$?
if [[ "$STATUS_GPU0" -ne 0 || "$STATUS_GPU1" -ne 0 ]]; then
  echo "Depth workers failed: GPU0=$STATUS_GPU0 GPU1=$STATUS_GPU1" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u scripts/probing/train_depth_probes.py \
  --output-root "$OUTPUT_ROOT" --sample-indices "$SAMPLES" \
  --model-labels internvl3_8b_base_presft --feature-levels "$LEVELS" \
  --probe-subdir probes --result-stem internvl3_8b_base_depth \
  --epochs 50 --batch-size 32 --lr 0.001 --early-stop-patience 10 \
  --probe-seed 0 --device cpu --skip-existing \
  > "$OUTPUT_ROOT/depth_probes_aggregate.log" 2>&1

"$PYTHON" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
label = "internvl3_8b_base_presft"
layers = "visual_output fusion_output projected_features layer_0 layer_1 layer_2 layer_3 layer_6 layer_9 layer_12 layer_15 layer_18 layer_21 layer_24 layer_27".split()
missing = [level for level in layers if not (root / "probes" / label / level / "metrics.json").is_file()]
if missing:
    raise SystemExit(f"Missing final weak-depth metrics: {missing}")
summary = root / "logme_common7" / label / "summary.json"
if not summary.is_file():
    raise SystemExit("Missing common-seven LogME summary")
print(json.dumps({"status": "complete", "depth_levels": len(layers), "logme": str(summary)}))
' "$OUTPUT_ROOT"
