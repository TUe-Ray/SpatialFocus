#!/usr/bin/env bash
# Thin VSI-Bench wrapper for controlled pre-projector patch-only cross-attention.
set -euo pipefail

export CONTROLLED_FUSION_ID=A_prime
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "$SCRIPT_DIR/eval_controlled_fusion_variant_vsibench.sh"
