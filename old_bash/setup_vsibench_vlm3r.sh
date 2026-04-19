#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_vsibench_vlm3r.sh /path/to/VLM-3R
# Example:
#   bash setup_vsibench_vlm3r.sh ~/VLM-3R

REPO_DIR="${1:-$HOME/VLM-3R}"
ENV_NAME="vsibench"
PY_VER="3.10"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[ERROR] Repo dir not found: $REPO_DIR" >&2
  exit 1
fi

# Optional on Leonardo: match the repo's CUDA 12.1 toolchain for builds
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.1 || true
fi

# Start clean
conda deactivate >/dev/null 2>&1 || true
conda remove -n "$ENV_NAME" --all -y >/dev/null 2>&1 || true

conda create -n "$ENV_NAME" python=${PY_VER} -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

# Match VLM-3R evaluation README: torch 2.1.1 + cu121
conda install -y \
  pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 \
  -c pytorch -c nvidia

cd "$REPO_DIR/thinking-in-space"

# Core eval deps
python -m pip install -e .
python -m pip install "s2wrapper @ git+https://github.com/bfshi/scaling_on_scales"
python -m pip install \
  transformers==4.40.0 \
  peft==0.10.0 \
  accelerate==0.29.1 \
  "huggingface_hub[hf_xet]"

# IMPORTANT:
# Skip google-generativeai / google-genai here.
# They are not needed for local VLM-3R VSiBench evaluation and often re-introduce
# protobuf dependency conflicts into this env.

# FlashAttention wheel from repo README. If it fails, keep going and use sdpa/eager.
python -m pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  || echo "[WARN] flash-attn install failed; you can still run with attn_implementation=sdpa or eager"

# Helpful sanity checks
python - <<'PY'
import torch
print('torch', torch.__version__)
print('torch cuda build', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
print('device count', torch.cuda.device_count())
PY

python -m pip check || true

echo
echo "[DONE] Environment '$ENV_NAME' is ready."
echo "Next suggestions:"
echo "  1) Run your eval with attn_implementation=sdpa first."
echo "  2) Do NOT install google-generativeai / google-genai into this env unless you specifically need Gemini-backed eval."
echo "  3) If flash-attn installed cleanly and GPU init is healthy, then you can try flash_attention_2 later."
