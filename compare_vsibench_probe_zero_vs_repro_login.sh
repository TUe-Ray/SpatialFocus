#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/vsibench_probe_spatial_ablation_common.sh
source "${REPO_ROOT}/scripts/vsibench_probe_spatial_ablation_common.sh"
vsibench_probe_ablation_init_defaults

echo "==== VSiBench zero-spatial vs Reproduction_2 login comparison ===="
echo "ZERO_OUTPUT_DIR=${ZERO_OUTPUT_DIR}"
echo "REPRO_OUTPUT_DIR=${REPRO_OUTPUT_DIR}"
echo "COMPARE_OUTPUT_DIR=${COMPARE_OUTPUT_DIR}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "SAMPLE_SEED=${SAMPLE_SEED}"
echo "OPTION_SHUFFLE_SEEDS=${OPTION_SHUFFLE_SEEDS}"
echo "==================================================================="

if vsibench_probe_should_check_outputs; then
  if ! vsibench_probe_check_eval_outputs "${ZERO_OUTPUT_DIR}"; then
    echo "Missing zero-spatial outputs. Please run:" >&2
    echo "sbatch submit_vsibench_probe_zero_spatial_dbg.slurm" >&2
    exit 1
  fi

  if ! vsibench_probe_check_eval_outputs "${REPRO_OUTPUT_DIR}"; then
    echo "Missing reproduction outputs. Please run:" >&2
    echo "sbatch submit_vsibench_probe_reproduction2_dbg.slurm" >&2
    exit 1
  fi
fi

cmd=(
  python "${REPO_ROOT}/scripts/compare_vsibench_probe_runs.py"
  --runs "${ZERO_OUTPUT_DIR}" "${REPRO_OUTPUT_DIR}"
  --output "${COMPARE_OUTPUT_DIR}"
)

vsibench_probe_print_command "${cmd[@]}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${cmd[@]}"
fi

if vsibench_probe_should_check_outputs; then
  if ! vsibench_probe_check_compare_outputs "${COMPARE_OUTPUT_DIR}"; then
    echo "[ERROR] Comparison did not produce all required outputs." >&2
    exit 1
  fi
fi

echo "Comparison written to ${COMPARE_OUTPUT_DIR}"
