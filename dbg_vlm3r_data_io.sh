#!/bin/bash
#SBATCH --job-name=DBGvlm3r_data_io
#SBATCH --nodes=4
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/chore/%x_%j.out
#SBATCH --error=logs/chore/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

mkdir -p logs/chore

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"
FAST_VLM3R="${FAST_VLM3R:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
WORK_VLM3R="${WORK_VLM3R:-/leonardo_work/EUHPC_D32_006/train_data/vlm3r}"
REPEAT="${REPEAT:-3}"
READ_BYTES="${READ_BYTES:-4194304}"

module load cuda/12.1
module load cudnn
module load profile/deeplrn

export PATH="$WORK/miniconda3/bin:$PATH"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u

ROOTS=("$FAST_VLM3R")
if [[ -d "$WORK_VLM3R" ]]; then
    ROOTS+=("$WORK_VLM3R")
fi

echo "[DBG_IO] job_id=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
echo "[DBG_IO] roots=${ROOTS[*]}"
echo "[DBG_IO] repeat=$REPEAT read_bytes=$READ_BYTES"

srun --kill-on-bad-exit=1 --wait=30 --export=ALL \
    python scripts/diagnose_vlm3r_data_io.py \
        --roots "${ROOTS[@]}" \
        --repeat "$REPEAT" \
        --read-bytes "$READ_BYTES" \
        --decord \
        --torch-load-one-sidecar
