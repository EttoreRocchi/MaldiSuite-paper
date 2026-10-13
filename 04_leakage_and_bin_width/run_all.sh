#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
$PY leakage_ladder.py
$PY bin_width_sensitivity.py
echo "ALL DONE $(date)"
