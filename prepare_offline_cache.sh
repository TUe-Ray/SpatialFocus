#!/usr/bin/env bash
set -euo pipefail

# Run this on a login node WITH internet access.
# It pre-downloads all model/dataset assets needed by offline compute nodes.

FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
HF_HOME="${HF_HOME:-$FAST_ROOT/hf_cache}"
MODEL_ROOT="${MODEL_ROOT:-$FAST_ROOT/hf_models/VLM3R}"
HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "$HF_HOME" "$MODEL_ROOT"
mkdir -p "$HF_HOME/hub" "$HF_HOME/datasets" "$HF_HOME/transformers" "$HF_HOME/modules"

export HF_HOME
export FAST_ROOT
export MODEL_ROOT
export HF_TOKEN
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# Keep this aligned with HF_HUB_CACHE so transformers offline lookups
# and snapshot_download read from the same place.
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"

echo "=== Prefetch Configuration ==="
echo "FAST_ROOT=$FAST_ROOT"
echo "HF_HOME=$HF_HOME"
echo "MODEL_ROOT=$MODEL_ROOT"
if [[ -n "$HF_TOKEN" ]]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is empty (OK for public repos)"
fi

python - <<'PY'
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


hf_token = os.environ.get("HF_TOKEN") or None
hf_home = Path(os.environ.get("HF_HOME", "/leonardo_scratch/fast/EUHPC_D32_006/hf_cache"))
model_root = Path(os.environ.get("MODEL_ROOT", "/leonardo_scratch/fast/EUHPC_D32_006/hf_models/VLM3R"))

repos = [
    ("Journey9ni/vlm-3r-llava-qwen2-lora", "model", Path("Journey9ni/vlm-3r-llava-qwen2-lora")),
    ("lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", "model", Path("LLaVA-NeXT-Video-7B-Qwen2")),
    ("google/siglip-so400m-patch14-384", "model", Path("siglip-so400m-patch14-384")),
]

print("\\n=== Downloading model repos ===")
for repo_id, repo_type, rel_target in repos:
    path = snapshot_download(repo_id=repo_id, repo_type=repo_type, token=hf_token)
    target = model_root / rel_target
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"[KEEP] {target} already exists")
    else:
        target.symlink_to(path)
        print(f"[LINK] {target} -> {path}")

print("\\n=== Downloading dataset repo snapshot ===")
ds_snapshot = snapshot_download(repo_id="nyu-visionx/VSI-Bench", repo_type="dataset", token=hf_token)
print(f"[OK] dataset snapshot: {ds_snapshot}")

print("\\n=== Building datasets cache artifacts ===")
dataset_kwargs = {"cache_dir": str(hf_home / "vsibench")}
if hf_token:
    dataset_kwargs["token"] = hf_token

ds = load_dataset("nyu-visionx/VSI-Bench", **dataset_kwargs)
print(ds)
for split in ds:
    print(f"[OK] split={split} rows={len(ds[split])}")

print("\\n=== Prefetch complete ===")
PY

echo "Done. You can now use offline compute nodes with HF_*_OFFLINE=1."
