#!/bin/bash
#SBATCH --job-name=vsibench_repro_dbg
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

echo "==== VSiBench Reproduction_2 probe debug job ===="
echo "REPRO_RUN_NAME=${REPRO_RUN_NAME}"
echo "REPRO_CHECKPOINT=${REPRO_CHECKPOINT}"
echo "REPRO_MODEL_ARGS=${REPRO_MODEL_ARGS}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "SAMPLE_SEED=${SAMPLE_SEED}"
echo "OPTION_SHUFFLE_SEEDS=${OPTION_SHUFFLE_SEEDS}"
echo "REPRO_OUTPUT_DIR=${REPRO_OUTPUT_DIR}"
echo "================================================="

cmd=(
  env
  "RUN_NAME=${REPRO_RUN_NAME}"
  "OUTPUT_DIR=${REPRO_OUTPUT_DIR}"
  "MODEL=${REPRO_MODEL}"
  "MODEL_ARGS=${REPRO_MODEL_ARGS}"
  "CHECKPOINT=${REPRO_CHECKPOINT}"
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
  if ! vsibench_probe_check_eval_outputs "${REPRO_OUTPUT_DIR}"; then
    echo "[ERROR] Reproduction probe did not produce all required outputs." >&2
    exit 1
  fi
fi

echo "Reproduction probe written to ${REPRO_OUTPUT_DIR}"
