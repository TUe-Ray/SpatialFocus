#!/bin/bash
#SBATCH --job-name=eval_vsibench_probe_200
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
NUM_SAMPLES="${NUM_SAMPLES:-200}"
# Under sbatch, BASH_SOURCE can point to a slurmd spool copy (often not writable).
# Prefer the submit directory so default outputs stay in user space.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
LMMS_ROOT="${REPO_ROOT}/thinking-in-space"

# Example:
# MODEL_ARGS="pretrained=/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2,model_base=/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32" \
# CHECKPOINT="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2" \
# RUN_NAME="selective_fusion_option_shuffle_200" \
# NUM_SAMPLES=200 \
# SAMPLE_SEED=42 \
# PROMPT_VARIANT=option_shuffle \
# OPTION_SHUFFLE_SEEDS=0,1,2 \
# bash eval_vsibench_probe.sh

PROMPT_VARIANT="${PROMPT_VARIANT:-option_shuffle}"

SAMPLE_SEED="${SAMPLE_SEED:-42}"
if [[ -z "${OPTION_SHUFFLE_SEEDS:-}" ]]; then
  if [[ -n "${OPTION_SHUFFLE_SEED:-}" ]]; then
    OPTION_SHUFFLE_SEEDS="${OPTION_SHUFFLE_SEED}"
  else
    OPTION_SHUFFLE_SEEDS="0,1,2"
  fi
fi
OPTION_SHUFFLE_SEED="${OPTION_SHUFFLE_SEED:-${OPTION_SHUFFLE_SEEDS%%,*}}"
RUN_NAME="${RUN_NAME:-vsibench_probe_${PROMPT_VARIANT}_${NUM_SAMPLES}_seed${SAMPLE_SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R}"
JOB_NAME="${SLURM_JOB_NAME:-${RUN_NAME}}"
JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${JOB_NAME}_${JOB_ID}}"
RAW_OUTPUT_DIR="${OUTPUT_DIR}/raw_lmms_eval"

MODEL="${MODEL:-vlm_3r}"
CHECKPOINT="${CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/Reproduction_2}"
MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=${CHECKPOINT},model_base=${MODEL_BASE},model_name=${MODEL_NAME},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
NOTES="${NOTES:-}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vsibench}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
HF_HOME="${HF_HOME:-${FAST_ROOT}/hf_cache}"

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

if command -v module >/dev/null 2>&1; then
  module purge || true
  unset LD_LIBRARY_PATH || true
  module load 2023 CUDA/12.1.1 || echo "[WARN] module load 2023 CUDA/12.1.1 failed; continuing"
fi

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  set -u
fi

if command -v conda >/dev/null 2>&1; then
  set +u
  conda activate "${CONDA_ENV}"
  set -u
else
  echo "[WARN] conda not found; continuing without conda activation" >&2
fi

export PROMPT_VARIANT
export NUM_SAMPLES
export SAMPLE_SEED
export OPTION_SHUFFLE_SEED
export OPTION_SHUFFLE_SEEDS
export RUN_NAME
export VSIBENCH_PROBE_OUTPUT_DIR="${OUTPUT_DIR}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLM3R_CODE_ROOT="${REPO_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ -z "${CUDA_HOME:-}" && -d "/leonardo/prod/opt/compilers/cuda/12.1/none" ]]; then
  export CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.1/none"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ "${LMMS_EVAL_LAUNCHER}" == "accelerate" ]]; then
  if command -v accelerate >/dev/null 2>&1; then
    ACCELERATE_CMD=(accelerate launch)
  elif python -c "import accelerate" >/dev/null 2>&1; then
    ACCELERATE_CMD=(python -m accelerate.commands.launch)
  else
    echo "[ERROR] Accelerate launcher selected, but no 'accelerate' command/module is available." >&2
    echo "[ERROR] Activate the correct env or set LMMS_EVAL_LAUNCHER=python." >&2
    exit 2
  fi
fi

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
  "${ACCELERATE_CMD[@]}" \
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
