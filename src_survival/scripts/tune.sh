#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/survival-simulator [hours] [output-dir]" >&2
  exit 2
fi
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
simulator="$1"
hours="${2:-5}"
output="${3:-$project_dir/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$project_dir"
exec .venv/bin/survival-tune \
  --simulator "$simulator" \
  --hours "$hours" \
  --out "$output"
