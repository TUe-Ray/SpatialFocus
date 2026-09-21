#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

[[ "$(hostname -f)" == *.snellius.surf.nl ]] || {
  echo "This preparation script is Snellius-only." >&2
  exit 2
}
[[ -d "$SOURCE_ROOT" ]] || {
  echo "Missing source cache: $SOURCE_ROOT" >&2
  exit 2
}
[[ ! -e "$FINAL_ROOT" ]] || {
  echo "Final cache already exists: $FINAL_ROOT" >&2
  exit 2
}

mkdir -p "$STAGING_ROOT" "$ARTIFACTS_ROOT" "$LOG_ROOT"

source_count="$(find "$SOURCE_ROOT" -mindepth 2 -maxdepth 2 -type f -name '*.pt' | wc -l)"
[[ "$source_count" == "2693" ]] || {
  echo "Source cache count is $source_count, expected 2693" >&2
  exit 2
}

cat <<EOF
Prepared L23 conversion paths.
source_count=$source_count
source_root=$SOURCE_ROOT
staging_root=$STAGING_ROOT
final_root=$FINAL_ROOT
artifacts_root=$ARTIFACTS_ROOT
smoke_command=sbatch $SCRIPT_DIR/smoke.sbatch
full_command=sbatch $SCRIPT_DIR/full.sbatch
EOF
