#!/usr/bin/env bash
# Wait for existing local GPU work, then smoke and run A-prime LogME end to end.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
BLOCKING_UNIT="${BLOCKING_UNIT:-spatialfocus-additional-presft-full.service}"
OUTPUT_DIR="$REPO_ROOT/logs/pre_sft_logme_proxy_a_prime_common7"
QUEUE_LOG="$OUTPUT_DIR/queue.log"
mkdir -p "$OUTPUT_DIR"

while systemctl --user is-active --quiet "$BLOCKING_UNIT"; do
  echo "[WAIT] $(date --iso-8601=seconds) $BLOCKING_UNIT remains active" >>"$QUEUE_LOG"
  sleep 60
done

idle_checks=0
while (( idle_checks < 3 )); do
  mapfile -t used_mib < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ "${#used_mib[@]}" -eq 2 && "${used_mib[0]}" -lt 2048 && "${used_mib[1]}" -lt 2048 ]]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  echo "[WAIT] $(date --iso-8601=seconds) idle_check=$idle_checks/3 gpu_mib=${used_mib[*]:-unknown}" >>"$QUEUE_LOG"
  sleep 20
done

echo "[RUN] $(date --iso-8601=seconds) A-prime one-video smoke" >>"$QUEUE_LOG"
bash "$REPO_ROOT/scripts/probing/run_pre_sft_a_prime_logme_local.sh" smoke >>"$QUEUE_LOG" 2>&1
echo "[RUN] $(date --iso-8601=seconds) A-prime full extraction and Common-7 LogME" >>"$QUEUE_LOG"
bash "$REPO_ROOT/scripts/probing/run_pre_sft_a_prime_logme_local.sh" full >>"$QUEUE_LOG" 2>&1
echo "[COMPLETE] $(date --iso-8601=seconds) A-prime Common-7 LogME" >>"$QUEUE_LOG"
