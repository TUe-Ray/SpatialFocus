#!/bin/bash
#SBATCH --job-name=eval_vsibench_probe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_ROOT="${REPO_ROOT}/thinking-in-space"

# Example:
# MODEL_ARGS="pretrained=/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2,model_base=/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32" \
# CHECKPOINT="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2" \
# RUN_NAME="selective_fusion_option_shuffle_100" \
# NUM_SAMPLES=100 \
# SAMPLE_SEED=42 \
# PROMPT_VARIANT=option_shuffle \
# OPTION_SHUFFLE_SEEDS=0,1,2 \
# bash eval_vsibench_probe.sh

PROMPT_VARIANT="${PROMPT_VARIANT:-option_shuffle}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
OPTION_SHUFFLE_SEED="${OPTION_SHUFFLE_SEED:-0}"
OPTION_SHUFFLE_SEEDS="${OPTION_SHUFFLE_SEEDS:-${OPTION_SHUFFLE_SEED}}"
RUN_NAME="${RUN_NAME:-vsibench_probe_${PROMPT_VARIANT}_${NUM_SAMPLES}_seed${SAMPLE_SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vsibench_probe/${RUN_NAME}}"
RAW_OUTPUT_DIR="${OUTPUT_DIR}/raw_lmms_eval"

MODEL="${MODEL:-vlm_3r}"
CHECKPOINT="${CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2}"
MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=${CHECKPOINT},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
NOTES="${NOTES:-}"

if [[ "${MODEL_ARGS}" == *"..."* ]]; then
  echo "[ERROR] MODEL_ARGS still contains placeholder '...'. Please set real pretrained/model_base values." >&2
  exit 2
fi

# If checkpoint is not explicitly set, infer it from pretrained=... in MODEL_ARGS.
if [[ -z "${CHECKPOINT}" || "${CHECKPOINT}" == "..." ]]; then
  PRETRAINED_FROM_ARGS="$(sed -n 's/.*pretrained=\([^,]*\).*/\1/p' <<< "${MODEL_ARGS}")"
  if [[ -n "${PRETRAINED_FROM_ARGS}" ]]; then
    CHECKPOINT="${PRETRAINED_FROM_ARGS}"
  fi
fi

mkdir -p "${OUTPUT_DIR}" "${RAW_OUTPUT_DIR}"

export PROMPT_VARIANT
export NUM_SAMPLES
export SAMPLE_SEED
export OPTION_SHUFFLE_SEED
export OPTION_SHUFFLE_SEEDS
export RUN_NAME
export VSIBENCH_PROBE_OUTPUT_DIR="${OUTPUT_DIR}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"

if [[ "${PROMPT_VARIANT}" == "evidence_json" ]]; then
  GEN_KWARGS="${GEN_KWARGS:-temperature=0,do_sample=false,top_p=1.0,num_beams=1,max_new_tokens=128}"
elif [[ "${PROMPT_VARIANT}" == "option_shuffle" ]]; then
  GEN_KWARGS="${GEN_KWARGS:-temperature=0,do_sample=false,top_p=1.0,num_beams=1,max_new_tokens=16}"
else
  echo "Unsupported PROMPT_VARIANT: ${PROMPT_VARIANT}" >&2
  exit 2
fi

cd "${LMMS_ROOT}"

if [[ "${LMMS_EVAL_LAUNCHER}" == "python" ]]; then
  python -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench_probe \
    --batch_size "${BATCH_SIZE}" \
    --gen_kwargs "${GEN_KWARGS}" \
    --log_samples \
    --log_samples_suffix "${RUN_NAME}" \
    --output_path "${RAW_OUTPUT_DIR}"
else
  accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench_probe \
    --batch_size "${BATCH_SIZE}" \
    --gen_kwargs "${GEN_KWARGS}" \
    --log_samples \
    --log_samples_suffix "${RUN_NAME}" \
    --output_path "${RAW_OUTPUT_DIR}"
fi

cd "${REPO_ROOT}"

IFS=", " read -r -a SHUFFLE_SEED_ARRAY <<< "${OPTION_SHUFFLE_SEEDS}"

python scripts/analyze_vsibench_probe.py \
  --run-dir "${OUTPUT_DIR}" \
  --raw-lmms-eval-dir "${RAW_OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --prompt-variant "${PROMPT_VARIANT}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-seed "${SAMPLE_SEED}" \
  --option-shuffle-seeds "${SHUFFLE_SEED_ARRAY[@]}" \
  --model "${MODEL}" \
  --model-args "${MODEL_ARGS}" \
  --model-name-or-path "${CHECKPOINT}" \
  --checkpoint "${CHECKPOINT}" \
  --notes "${NOTES}"

python scripts/generate_vsibench_probe_report.py --run-dir "${OUTPUT_DIR}"

echo "VSiBench probe artifacts written to ${OUTPUT_DIR}"
