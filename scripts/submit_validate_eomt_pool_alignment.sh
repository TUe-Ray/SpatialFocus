#!/bin/bash
#SBATCH --job-name=validate_eomt_pool
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/validate/%x_%j.out
#SBATCH --error=logs/validate/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$PROJECT_ROOT/logs/validate"

bash "$PROJECT_ROOT/scripts/run_validate_eomt_pool_alignment.sh"
