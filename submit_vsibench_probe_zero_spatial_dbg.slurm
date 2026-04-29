#!/bin/bash
#SBATCH --job-name=vsibench_zero_dbg
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "${REPO_ROOT}"
mkdir -p logs/slurm

# Environment activation is delegated to eval_vsibench_probe.sh, which uses the
# same module/conda setup as the existing VLM-3R probe eval workflow.
# TODO: activate environment here if your local cluster setup requires it before calling eval_vsibench_probe.sh.

# shellcheck source=scripts/vsibench_probe_spatial_ablation_common.sh
source "${REPO_ROOT}/scripts/vsibench_probe_spatial_ablation_common.sh"
vsibench_probe_ablation_init_defaults

echo "==== VSiBench zero-spatial probe debug job ===="
echo "ZERO_RUN_NAME=${ZERO_RUN_NAME}"
echo "ZERO_CHECKPOINT=${ZERO_CHECKPOINT}"
echo "ZERO_MODEL_ARGS=${ZERO_MODEL_ARGS}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "SAMPLE_SEED=${SAMPLE_SEED}"
echo "OPTION_SHUFFLE_SEEDS=${OPTION_SHUFFLE_SEEDS}"
echo "ZERO_OUTPUT_DIR=${ZERO_OUTPUT_DIR}"
echo "==============================================="

cmd=(
  env
  "RUN_NAME=${ZERO_RUN_NAME}"
  "OUTPUT_DIR=${ZERO_OUTPUT_DIR}"
  "MODEL=${ZERO_MODEL}"
  "MODEL_ARGS=${ZERO_MODEL_ARGS}"
  "CHECKPOINT=${ZERO_CHECKPOINT}"
  "NUM_SAMPLES=${NUM_SAMPLES}"
  "SAMPLE_SEED=${SAMPLE_SEED}"
  "PROMPT_VARIANT=option_shuffle"
  "OPTION_SHUFFLE_SEEDS=${OPTION_SHUFFLE_SEEDS}"
  bash "${REPO_ROOT}/eval_vsibench_probe.sh"
)

vsibench_probe_print_command "${cmd[@]}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${cmd[@]}"
fi

if vsibench_probe_should_check_outputs; then
  if ! vsibench_probe_check_eval_outputs "${ZERO_OUTPUT_DIR}"; then
    echo "[ERROR] Zero-spatial probe did not produce all required outputs." >&2
    exit 1
  fi
fi

echo "Zero-spatial probe written to ${ZERO_OUTPUT_DIR}"
