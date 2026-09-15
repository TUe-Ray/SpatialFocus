#!/usr/bin/env bash
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/shuang/proxy_outputs/legacy_pre_sft_completion_a100_v1/logs/slurm_%x_%j.out

set -euo pipefail
REPO_ROOT="${SPATIALFOCUS_ROOT:-/home/shuang/SpatialFocus}"
export SPATIALFOCUS_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec bash "$REPO_ROOT/scripts/probing/run_legacy_pre_sft_completion_a100.sh" summarize
