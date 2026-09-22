#!/usr/bin/env bash
# Submit with candidate_a|candidate_b as the script argument and an explicit --output log path.
# No account or QoS is inferred from another Snellius user/project.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00

set -euo pipefail
CANDIDATE="${CANDIDATE:-${1:-}}"
: "${CANDIDATE:?Pass candidate_a or candidate_b as the script argument}"
case "$CANDIDATE" in
  candidate_a|candidate_b) ;;
  *) echo "Unsupported candidate: $CANDIDATE" >&2; exit 2 ;;
esac

REPO_ROOT="${SPATIALFOCUS_ROOT:-/home/shuang/SpatialFocus}"
SOURCE_ROOT="${QWEN35_SOURCE_ROOT:-/scratch-shared/shuang/SpatialStack-Qwen35-VGGT-d37ed60}"
ENV_ROOT="${QWEN35_ENV_ROOT:-/scratch-shared/shuang/qwen35_proxy_env}"
OUTPUT_ROOT="${QWEN35_PROXY_OUTPUT_ROOT:-/home/shuang/proxy_outputs/pre_sft_zero_cost_proxies_a100_v1/qwen35_cached_vggt_ab_v1}"

module load 2025
module load CUDA/12.9.1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SOURCE_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch-shared/shuang/qwen35_hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$SOURCE_ROOT/scripts/profiling/fla_configs_a100"

echo "QWEN35_PROXY job=$SLURM_JOB_ID candidate=$CANDIDATE host=$(hostname -f)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$ENV_ROOT/bin/python" "$SOURCE_ROOT/scripts/train/validate_qwen35_runtime.py" \
  --attention-implementation flash_attention_2 --require-fast-linear-attention --require-cuda
"$ENV_ROOT/bin/python" -u "$REPO_ROOT/scripts/probing/run_qwen35_cached_vggt_proxy_a100.py" \
  --mode run --candidate "$CANDIDATE" --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
