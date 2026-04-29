#!/bin/bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

NUM_SAMPLES="${NUM_SAMPLES:-300}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PROMPT_VARIANT="${PROMPT_VARIANT:-option_shuffle}"
OPTION_SHUFFLE_SEEDS="${OPTION_SHUFFLE_SEEDS:-0,1,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/vsibench_probe}"

MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

REPRO_MODEL="${REPRO_MODEL:-vlm_3r}"
REPRO_CHECKPOINT="${REPRO_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2}"
REPRO_MODEL_ARGS="${REPRO_MODEL_ARGS:-pretrained=${REPRO_CHECKPOINT},model_base=${MODEL_BASE},model_name=${MODEL_NAME},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM}}"

ZERO_MODEL="${ZERO_MODEL:-vlm_3r}"
ZERO_CHECKPOINT="${ZERO_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/ablation_no_spatial_25p_39895083}"
ZERO_MODEL_ARGS="${ZERO_MODEL_ARGS:-pretrained=${ZERO_CHECKPOINT},model_base=${MODEL_BASE},model_name=${MODEL_NAME},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},zero_spatial_features=True}"

IFS=", " read -r -a SHUFFLE_SEED_ARRAY <<< "${OPTION_SHUFFLE_SEEDS}"
SEED_COUNT="${#SHUFFLE_SEED_ARRAY[@]}"

ZERO_RUN_NAME="${ZERO_RUN_NAME:-zero_spatial_option_shuffle_${NUM_SAMPLES}_seed${SAMPLE_SEED}_${SEED_COUNT}seeds}"
REPRO_RUN_NAME="${REPRO_RUN_NAME:-reproduction2_option_shuffle_${NUM_SAMPLES}_seed${SAMPLE_SEED}_${SEED_COUNT}seeds}"
ZERO_OUTPUT_DIR="${ZERO_OUTPUT_DIR:-${OUTPUT_ROOT}/${ZERO_RUN_NAME}}"
REPRO_OUTPUT_DIR="${REPRO_OUTPUT_DIR:-${OUTPUT_ROOT}/${REPRO_RUN_NAME}}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${OUTPUT_ROOT}/compare_zero_spatial_vs_reproduction2_${NUM_SAMPLES}_seed${SAMPLE_SEED}_${SEED_COUNT}seeds}"

run_probe() {
  local run_name="$1"
  local output_dir="$2"
  local model="$3"
  local model_args="$4"
  local checkpoint="$5"

  local cmd=(
    env
    "RUN_NAME=${run_name}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "OUTPUT_DIR=${output_dir}"
    "MODEL=${model}"
    "MODEL_ARGS=${model_args}"
    "CHECKPOINT=${checkpoint}"
    "NUM_SAMPLES=${NUM_SAMPLES}"
    "SAMPLE_SEED=${SAMPLE_SEED}"
    "PROMPT_VARIANT=${PROMPT_VARIANT}"
    "OPTION_SHUFFLE_SEEDS=${OPTION_SHUFFLE_SEEDS}"
    bash "${REPO_ROOT}/eval_vsibench_probe.sh"
  )

  printf '[CMD] %q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${cmd[@]}"
  fi
}

compare_runs() {
  local cmd=(
    python "${REPO_ROOT}/scripts/compare_vsibench_probe_runs.py"
    --runs "${ZERO_OUTPUT_DIR}" "${REPRO_OUTPUT_DIR}"
    --output "${COMPARE_OUTPUT_DIR}"
  )

  printf '[CMD] %q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${cmd[@]}"
  fi
}

echo "==== VSiBench probe spatial ablation pair ===="
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "SAMPLE_SEED=${SAMPLE_SEED}"
echo "PROMPT_VARIANT=${PROMPT_VARIANT}"
echo "OPTION_SHUFFLE_SEEDS=${OPTION_SHUFFLE_SEEDS}"
echo "ZERO_RUN_NAME=${ZERO_RUN_NAME}"
echo "REPRO_RUN_NAME=${REPRO_RUN_NAME}"
echo "COMPARE_OUTPUT_DIR=${COMPARE_OUTPUT_DIR}"
echo "=============================================="

run_probe "${ZERO_RUN_NAME}" "${ZERO_OUTPUT_DIR}" "${ZERO_MODEL}" "${ZERO_MODEL_ARGS}" "${ZERO_CHECKPOINT}"
run_probe "${REPRO_RUN_NAME}" "${REPRO_OUTPUT_DIR}" "${REPRO_MODEL}" "${REPRO_MODEL_ARGS}" "${REPRO_CHECKPOINT}"
compare_runs

echo "Paired comparison written to ${COMPARE_OUTPUT_DIR}"
