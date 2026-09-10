#!/usr/bin/env bash
# Submit with: sbatch --partition=gpu_a100 --gpus=1 --ntasks=1 this-file
# PHASE may also be additional-smoke/additional-formal for explicitly separate
# controlled-fusion and frozen-VSI roster extensions.
# No account/QoS is guessed here.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:30:00

set -euo pipefail

PHASE="${PHASE:-smoke}"
REPO_ROOT="${SPATIALFOCUS_ROOT:-/home/shuang/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r_a100_proxy}"
OUTPUT_BASE="${PROXY_OUTPUT_ROOT:-/home/shuang/proxy_outputs}/pre_sft_zero_cost_proxies_a100_v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE/$PHASE}"
CANDIDATES="${CANDIDATES:-}"

export HF_HOME="${HF_HOME:-/home/shuang/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_BASE"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite output: $OUTPUT_ROOT" >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import accelerate, peft, torch, transformers
assert peft.__version__ == "0.4.0", peft.__version__
assert torch.__version__.startswith("2.1.1"), torch.__version__
print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__})
PY

EXTRA_ARGS=()
if [[ -n "$CANDIDATES" ]]; then
  EXTRA_ARGS+=(--candidates "$CANDIDATES")
fi

conda run --no-capture-output -n "$ENV_NAME" python -u \
  "$REPO_ROOT/scripts/probing/run_pre_sft_trainable_proxy_a100.py" \
  --mode "$PHASE" --output-root "$OUTPUT_ROOT" \
  --base-model /home/shuang/models/base/LLaVA-NeXT-Video-7B-Qwen2 \
  --siglip-model /home/shuang/models/base/siglip-so400m-patch14-384 \
  --forward-frames-root /home/shuang/probing_data/forward_frames_32_v1 \
  --probe-targets-root /home/shuang/probing_data/probe_targets_empty \
  --feature-root /home/shuang/probing_data/cut3r_features \
  --data-yaml "$REPO_ROOT/scripts/probing/scannet_depth_probe_snellius_data.yaml" \
  --sample-indices /home/shuang/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json \
  --c1-root /home/shuang/c1_artifacts \
  --controlled-c1-root /home/shuang/c1_artifacts/controlled_fusion_pre_sft_v1/c1 \
  --geometry-c1-root /home/shuang/c1_artifacts/c1_geometry_pre_sft_v1 \
  --geometry-feature-root /home/shuang/probing_data/cut3r_point_maps_32_v1 \
  --eomt-cache-root /home/shuang/probing_data/eomt_consumer_grid_v2 \
  --device cuda:0 --dtype float16 --rng-seed 42 "${EXTRA_ARGS[@]}"
