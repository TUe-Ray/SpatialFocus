#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_JSON="${OUTPUT_JSON:-$PROJECT_ROOT/logs/eomt_object_block_phase1_report.json}"

cd "$PROJECT_ROOT"

echo "[INFO] Running Phase-1 EoMT object-block validation script"
echo "[INFO] Output JSON: $OUTPUT_JSON"

python scripts/validate_eomt_object_block_phase1.py --output_json "$OUTPUT_JSON"

echo "[INFO] Done"
