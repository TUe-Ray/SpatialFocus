#!/usr/bin/env bash
# Wait for the unrelated controlled post-SFT depth run, then use the two GPUs
# sequentially for the missing pre-SFT object-token and Visual geo-RoPE caches.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
export PATH="/home/shaoruei/miniconda3/bin:${PATH:-}"
if [[ -f /home/shaoruei/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/shaoruei/miniconda3/etc/profile.d/conda.sh
fi
BLOCKING_UNIT="${BLOCKING_UNIT:-spatialfocus-controlled-fusion-post-sft-depth.service}"
LOG_DIR="$REPO_ROOT/logs/pre_sft_logme_proxy_remaining_diagnostics_v1"
QUEUE_LOG="$LOG_DIR/queue.log"
mkdir -p "$LOG_DIR"

while systemctl --user is-active --quiet "$BLOCKING_UNIT"; do
  echo "[WAIT] $BLOCKING_UNIT remains active" >>"$QUEUE_LOG"
  sleep 60
done

# Require three consecutive idle checks so a short hand-off between other jobs
# cannot be mistaken for permission to start a new sharded model process.
idle_checks=0
while (( idle_checks < 3 )); do
  mapfile -t used_mib < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ "${#used_mib[@]}" -eq 2 && "${used_mib[0]}" -lt 2048 && "${used_mib[1]}" -lt 2048 ]]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  sleep 20
done

echo "[RUN] Baseline+depth and SS+depth loss-only forward-equivalence smoke" >>"$QUEUE_LOG"
bash "$REPO_ROOT/scripts/probing/run_pre_sft_loss_only_logme_equivalence_local.sh" >>"$QUEUE_LOG" 2>&1

echo "[RUN] Extra Object Token full pre-SFT extraction + Common-7 LogME" >>"$QUEUE_LOG"
bash "$REPO_ROOT/scripts/probing/run_pre_sft_logme_remaining_recache_local.sh" extra_object_token >>"$QUEUE_LOG" 2>&1

echo "[RUN] Visual geo-RoPE one-video smoke" >>"$QUEUE_LOG"
bash "$REPO_ROOT/scripts/probing/run_c1_geometry_pre_sft_depth_probe_local.sh" smoke visual_geo_rope >>"$QUEUE_LOG" 2>&1

echo "[RUN] Visual geo-RoPE full pre-SFT extraction + Common-7 LogME" >>"$QUEUE_LOG"
bash "$REPO_ROOT/scripts/probing/run_pre_sft_logme_remaining_recache_local.sh" visual_geo_rope >>"$QUEUE_LOG" 2>&1

echo "[RUN] Build complete 17-model Common-7 inventory" >>"$QUEUE_LOG"
conda run --no-capture-output -n vlm3r python -u \
  "$REPO_ROOT/scripts/probing/build_complete_pre_sft_logme_inventory.py" >>"$QUEUE_LOG" 2>&1

echo "[COMPLETE] Remaining six-model LogME work and 17-model inventory finished" >>"$QUEUE_LOG"
