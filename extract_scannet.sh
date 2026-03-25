#!/bin/bash
#SBATCH -A EUHPC_D32_006
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH -J vlm3r_scannet_extract_1gpu
#SBATCH -o /leonardo/home/userexternal/shuang00/VLM-3R/logs/%x-%j.out
#SBATCH -e /leonardo/home/userexternal/shuang00/VLM-3R/logs/%x-%j.err

module purge
module load profile/deeplrn
module load cuda/12.1

source ~/.bashrc
conda activate vlm3r

export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.1/none
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

REPO=/leonardo/home/userexternal/shuang00/VLM-3R

python $REPO/scripts/extract_spatial_features.py \
  --input-dir /leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/videos \
  --output-dir /leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/spatial_features \
  --cut3r-weights-path $REPO/CUT3R/src/cut3r_512_dpt_4_64.pth \
  --processor-config-path $REPO/processor_config.json \
  --gpu-ids 0 \
  --batch-size 1 \
  --frames-upbound 32