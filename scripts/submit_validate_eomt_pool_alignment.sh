#!/bin/bash
#SBATCH --job-name=validate_eomt_pool
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --mem=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="/leonardo/home/userexternal/shuang00/VLM-3R/logs/validate"

# Create log directory before SLURM uses it
mkdir -p "$LOG_DIR"

# Redirect output to our log directory
exec > >(tee "$LOG_DIR/validate_eomt_pool_${SLURM_JOB_ID}.out")
exec 2> >(tee "$LOG_DIR/validate_eomt_pool_${SLURM_JOB_ID}.err" >&2)

bash "$PROJECT_ROOT/scripts/run_validate_eomt_pool_alignment.sh"
