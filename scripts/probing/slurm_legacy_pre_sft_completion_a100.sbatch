#!/usr/bin/env bash
# One process on one full A100. Submit through submit_legacy_pre_sft_completion_a100.sh.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/home/shuang/proxy_outputs/legacy_pre_sft_completion_a100_v1/logs/slurm_%x_%j.out

set -euo pipefail

PHASE="${PHASE:?Set PHASE=smoke or PHASE=candidate}"
REPO_ROOT="${SPATIALFOCUS_ROOT:-/home/shuang/SpatialFocus}"
export SPATIALFOCUS_ROOT="$REPO_ROOT"
export HF_HOME="${HF_HOME:-/home/shuang/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/spatialfocus_mpl_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "hostname=$(hostname -f) slurm_job_id=$SLURM_JOB_ID phase=$PHASE candidate=${CANDIDATE:-none}"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
/home/shuang/miniconda3/envs/vlm3r_a100_proxy/bin/python - <<'PY'
import accelerate, peft, torch, transformers
assert peft.__version__ == "0.4.0", peft.__version__
assert torch.__version__.startswith("2.1.1"), torch.__version__
print({
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "transformers": transformers.__version__, "peft": peft.__version__,
    "accelerate": accelerate.__version__, "cuda_devices": torch.cuda.device_count(),
    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "tf32_cudnn": torch.backends.cudnn.allow_tf32,
})
assert torch.cuda.device_count() == 1
PY

if [[ "$PHASE" == smoke ]]; then
  exec bash "$REPO_ROOT/scripts/probing/run_legacy_pre_sft_completion_a100.sh" smoke
fi
if [[ "$PHASE" == candidate ]]; then
  exec bash "$REPO_ROOT/scripts/probing/run_legacy_pre_sft_completion_a100.sh" candidate "${CANDIDATE:?Set CANDIDATE}"
fi
echo "Unsupported PHASE=$PHASE" >&2
exit 2
