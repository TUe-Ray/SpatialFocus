#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_BASE="${CONDA_BASE:-$WORK/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"

HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D32_006/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

OUTPUT_JSON="${OUTPUT_JSON:-logs/eomt_pool_validation_report.json}"

cd "$PROJECT_ROOT"
mkdir -p logs

module purge || true
module load cuda/12.1 || true
module load cudnn || true
module load profile/deeplrn || true

export PATH="$CONDA_BASE/bin:$PATH"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u

if [[ -v LD_LIBRARY_PATH && -n "$LD_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
fi

export HF_HOME
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/validate_eomt_pool_alignment.py --output_json "$OUTPUT_JSON"

echo "Validation report saved to: $OUTPUT_JSON"
