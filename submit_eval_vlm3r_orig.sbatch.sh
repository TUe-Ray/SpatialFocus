#!/bin/bash
#SBATCH --job-name=0411_Eval_VLM3R_Orig
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

cd /leonardo/home/userexternal/shuang00/VLM-3R

export CONDA_ENV=vsibench
export NUM_PROCESSES=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LIMIT=0
export RUN_NAME=sbatch_orig_same_model

# Keep model paths explicit for parity.
export MODEL_ROOT=/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R
export PRETRAINED_LOCAL=$MODEL_ROOT/Journey9ni/vlm-3r-llava-qwen2-lora
export MODEL_BASE_LOCAL=$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2
export SIGLIP_LOCAL=$MODEL_ROOT/siglip-so400m-patch14-384

./run_vsi_leonardo_interactive.sh
