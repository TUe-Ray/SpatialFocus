#!/bin/bash

vsibench_probe_ablation_init_defaults() {
  if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "[ERROR] REPO_ROOT must be set before sourcing spatial ablation defaults." >&2
    return 2
  fi

  NUM_SAMPLES="${NUM_SAMPLES:-200}"
  SAMPLE_SEED="${SAMPLE_SEED:-42}"
  PROMPT_VARIANT="${PROMPT_VARIANT:-option_shuffle}"
  OPTION_SHUFFLE_SEEDS="${OPTION_SHUFFLE_SEEDS:-0,1,2}"

  if [[ "${PROMPT_VARIANT}" != "option_shuffle" ]]; then
    echo "[ERROR] This spatial ablation workflow only supports PROMPT_VARIANT=option_shuffle; got ${PROMPT_VARIANT}" >&2
    return 2
  fi

  IFS=", " read -r -a SHUFFLE_SEED_ARRAY <<< "${OPTION_SHUFFLE_SEEDS}"
  SEED_COUNT="${#SHUFFLE_SEED_ARRAY[@]}"
  if [[ "${SEED_COUNT}" -eq 0 ]]; then
    echo "[ERROR] OPTION_SHUFFLE_SEEDS must contain at least one seed." >&2
    return 2
  fi

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

  if [[ "${ZERO_MODEL_ARGS}" != *"zero_spatial_features=True"* ]]; then
    echo "[ERROR] ZERO_MODEL_ARGS must include zero_spatial_features=True for the ablation run." >&2
    return 2
  fi

  ZERO_RUN_NAME="${ZERO_RUN_NAME:-zero_spatial_option_shuffle_${NUM_SAMPLES}_seed${SAMPLE_SEED}_${SEED_COUNT}seeds}"
  REPRO_RUN_NAME="${REPRO_RUN_NAME:-reproduction2_option_shuffle_${NUM_SAMPLES}_seed${SAMPLE_SEED}_${SEED_COUNT}seeds}"
  COMPARE_RUN_NAME="${COMPARE_RUN_NAME:-compare_zero_spatial_vs_reproduction2_${NUM_SAMPLES}_seed${SAMPLE_SEED}_${SEED_COUNT}seeds}"

  ZERO_OUTPUT_DIR="${ZERO_OUTPUT_DIR:-${OUTPUT_ROOT}/${ZERO_RUN_NAME}}"
  REPRO_OUTPUT_DIR="${REPRO_OUTPUT_DIR:-${OUTPUT_ROOT}/${REPRO_RUN_NAME}}"
  COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${OUTPUT_ROOT}/${COMPARE_RUN_NAME}}"
}

vsibench_probe_should_check_outputs() {
  [[ "${DRY_RUN:-0}" != "1" || "${CHECK_OUTPUTS:-0}" == "1" ]]
}

vsibench_probe_print_command() {
  printf '[CMD] %q ' "$@"
  echo
}

vsibench_probe_require_files() {
  local missing=0
  local path
  for path in "$@"; do
    if [[ ! -f "${path}" ]]; then
      echo "[ERROR] Missing required file: ${path}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

vsibench_probe_check_eval_outputs() {
  local output_dir="$1"
  vsibench_probe_require_files \
    "${output_dir}/selected_samples.json" \
    "${output_dir}/predictions.jsonl" \
    "${output_dir}/stats.json" \
    "${output_dir}/sample_robustness.jsonl" \
    "${output_dir}/report.md"
}

vsibench_probe_check_compare_outputs() {
  local output_dir="$1"
  vsibench_probe_require_files \
    "${output_dir}/report.md" \
    "${output_dir}/stats.json" \
    "${output_dir}/paired_win_loss_by_question_type_rows.csv" \
    "${output_dir}/paired_win_loss_by_question_type_samples.csv" \
    "${output_dir}/paired_row_outcomes.jsonl" \
    "${output_dir}/paired_sample_outcomes.jsonl"
}
