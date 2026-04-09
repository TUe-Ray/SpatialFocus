#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_JSON="${OUTPUT_JSON:-$PROJECT_ROOT/logs/eomt_object_block_integration_report.json}"

cd "$PROJECT_ROOT"

echo "[INFO] Running sequence-level EoMT object-block integration validation"
echo "[INFO] Output JSON: $OUTPUT_JSON"

python scripts/validate_eomt_object_block_integration.py --output_json "$OUTPUT_JSON"

echo "[INFO] Done"
