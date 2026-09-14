#!/usr/bin/env bash
# Continue C/D/E/H only if the already-running B unit completed successfully.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PREDECESSOR="${PREDECESSOR_UNIT:-spatialfocus-controlled-logme-b.service}"
RUNNER="$REPO_ROOT/scripts/probing/run_controlled_fusion_logme_recache_local.sh"
LOG="$REPO_ROOT/logs/pre_sft_logme_proxy_controlled_fusion_v1/queue.log"
mkdir -p "$(dirname "$LOG")"

while systemctl --user is-active --quiet "$PREDECESSOR"; do
  printf '[WAIT] %s remains active\n' "$PREDECESSOR" >>"$LOG"
  sleep 30
done
RESULT="$(systemctl --user show "$PREDECESSOR" --property=Result --value)"
[[ "$RESULT" == "success" ]] || {
  printf '[STOP] %s result=%s; no subsequent candidate will start\n' "$PREDECESSOR" "$RESULT" >>"$LOG"
  exit 1
}

for identifier in C D E H; do
  printf '[RUN] %s\n' "$identifier" >>"$LOG"
  bash "$RUNNER" "$identifier" >>"$LOG" 2>&1
done
printf '[COMPLETE] C/D/E/H succeeded\n' >>"$LOG"
