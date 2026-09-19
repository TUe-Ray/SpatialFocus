#!/usr/bin/env bash

set -Eeuo pipefail

RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNTIME_DIR/config.sh"

module purge
module load 2023
module load CUDA/12.1.1

[[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || {
  echo "Missing environment: $VLM3R_CONDA_PREFIX" >&2
  exit 2
}
export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$STAGING_ROOT/_runtime_cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

vggt_runtime_preflight() {
  local source_commit
  local checkpoint_hash
  source_commit="$(git -C "$REPO_DIR/third_party/VGGT" rev-parse HEAD)"
  [[ "$source_commit" == "$VGGT_SOURCE_COMMIT" ]] || {
    echo "VGGT source mismatch: $source_commit" >&2
    return 2
  }
  checkpoint_hash="$(sha256sum "$VGGT_WEIGHTS/model.safetensors" | awk '{print $1}')"
  [[ "$checkpoint_hash" == "$VGGT_WEIGHTS_SHA256" ]] || {
    echo "VGGT checkpoint mismatch: $checkpoint_hash" >&2
    return 2
  }

  echo "run_timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname -f)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
  echo "repository_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "vggt_source_commit=$source_commit"
  echo "vggt_checkpoint=$VGGT_WEIGHTS/model.safetensors"
  echo "vggt_checkpoint_sha256=$checkpoint_hash"
  echo "processor_config=$PROCESSOR_CONFIG"
  echo "processor_config_sha256=$(sha256sum "$PROCESSOR_CONFIG" | awk '{print $1}')"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python - <<'PY'
from importlib.metadata import version
import torch

for name in ("torch", "torchvision", "transformers", "decord", "numpy", "Pillow", "safetensors", "huggingface_hub", "einops"):
    print(f"{name}={version(name)}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_device_count={torch.cuda.device_count()}")
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
PY
}

run_vggt_extract() {
  local input_mode="$1"
  local input_path="$2"
  local output_dir="$3"
  local -a command=(
    python "$EXTRACTOR"
    "$input_mode" "$input_path"
    --output-dir "$output_dir"
    --vggt-weights-path "$VGGT_WEIGHTS"
    --vggt-input-size 518
    --layer-indices 11,17,23
    --processor-config-path "$PROCESSOR_CONFIG"
    --gpu-ids 0
    --precision bf16
    --batch-size 1
    --frames-upbound 32
    --video-fps 1
  )
  printf 'EXACT_COMMAND='
  printf '%q ' "${command[@]}"
  printf '\n'
  "${command[@]}"
}
