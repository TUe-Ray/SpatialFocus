#!/bin/bash
#SBATCH --job-name=h1_diag_cut3r
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/diag/%x_%j.out
#SBATCH --error=logs/diag/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"

BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"
OURS_CHECKPOINT="${OURS_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/feat_align_cut3r_50p_40724606}"
MODEL_BASE="${MODEL_BASE:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_PATH="${SIGLIP_PATH:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"

DATA_PATH="${DATA_PATH:-$REPO_DIR/scripts/VLM_3R/vsibench_data.yaml}"
DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$DATA_ROOT}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features}"

OUT_ROOT="${OUT_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/diag/h1_cut3r}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
H1_DUMP="${H1_DUMP:-$OUT_ROOT/$RUN_NAME/h1_dump.pt}"
DIAG_OUT="${DIAG_OUT:-$OUT_ROOT/$RUN_NAME/diag}"

SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
MAX_SAMPLE_TRIES="${MAX_SAMPLE_TRIES:-200}"
FRAME="${FRAME:-0}"
ROI_INDEX="${ROI_INDEX:-105}"
PROBE_STEPS="${PROBE_STEPS:-300}"

if [[ -z "$BASELINE_CHECKPOINT" ]]; then
  echo "[ERROR] Please set BASELINE_CHECKPOINT to the CE-only checkpoint."
  echo "Example:"
  echo "  BASELINE_CHECKPOINT=/path/to/ce_only_ckpt sbatch dump_h1_diagnostic_cut3r.sh"
  exit 1
fi

mkdir -p logs/diag "$(dirname "$H1_DUMP")" "$DIAG_OUT"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  unset LD_LIBRARY_PATH || true
  module load 2023 CUDA/12.1.1 || module load cuda/12.1 || true
fi

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u
fi

set +u
conda activate "$CONDA_ENV"
set -u

export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D32_006/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLM3R_CODE_ROOT="$REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "BASELINE_CHECKPOINT=$BASELINE_CHECKPOINT"
echo "OURS_CHECKPOINT=$OURS_CHECKPOINT"
echo "H1_DUMP=$H1_DUMP"
echo "DIAG_OUT=$DIAG_OUT"

python scripts/dump_h1_diagnostic.py \
  --baseline-checkpoint "$BASELINE_CHECKPOINT" \
  --ours-checkpoint "$OURS_CHECKPOINT" \
  --model-base "$MODEL_BASE" \
  --siglip-path "$SIGLIP_PATH" \
  --cut3r-weights "$CUT3R_WEIGHTS" \
  --data-path "$DATA_PATH" \
  --image-folder "$DATA_ROOT" \
  --video-folder "$DATA_ROOT" \
  --spatial-features-root "$SPATIAL_FEATURES_ROOT" \
  --spatial-features-subdir "$SPATIAL_FEATURES_SUBDIR" \
  --sample-index "$SAMPLE_INDEX" \
  --max-sample-tries "$MAX_SAMPLE_TRIES" \
  --output "$H1_DUMP"

python scripts/spatial_rank_diagnostics.py \
  --input "$H1_DUMP" \
  --output-dir "$DIAG_OUT" \
  --pool-mode bilinear \
  --frame "$FRAME" \
  --roi-index "$ROI_INDEX" \
  --anchors-per-frame 128 \
  --positive-top-percent 10 \
  --negative-bottom-percent 30 \
  --probe-steps "$PROBE_STEPS" \
  --seed 42
