#!/usr/bin/env bash

set -Eeuo pipefail

RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNTIME_DIR/config.sh"

module purge
module load 2023

[[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || {
  echo "Missing environment: $VLM3R_CONDA_PREFIX" >&2
  exit 2
}

export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

vggt_l23_preflight() {
  [[ -d "$SOURCE_ROOT" ]] || {
    echo "Missing source cache: $SOURCE_ROOT" >&2
    return 2
  }
  [[ ! -e "$FINAL_ROOT" ]] || {
    echo "Refusing to overwrite existing final cache: $FINAL_ROOT" >&2
    return 2
  }
  mkdir -p "$STAGING_ROOT" "$ARTIFACTS_ROOT" "$LOG_ROOT"
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname -f)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
  echo "repository_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "source_root=$SOURCE_ROOT"
  echo "staging_root=$STAGING_ROOT"
  echo "final_root=$FINAL_ROOT"
  echo "artifacts_root=$ARTIFACTS_ROOT"
  python - <<'PY'
from importlib.metadata import version
import torch

for name in ("torch", "pandas", "pyarrow", "numpy"):
    print(f"{name}={version(name)}")
print(f"torch_cuda={torch.version.cuda}")
PY
}

run_vggt_l23_conversion() {
  local scope="$1"
  shift
  local -a command=(
    python "$CONVERTER"
    --scope "$scope"
    --source-root "$SOURCE_ROOT"
    --staging-root "$STAGING_ROOT"
    --final-root "$FINAL_ROOT"
    --artifacts-root "$ARTIFACTS_ROOT"
    "$@"
  )
  printf 'EXACT_COMMAND='
  printf '%q ' "${command[@]}"
  printf '\n'
  "${command[@]}"
}
