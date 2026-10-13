#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python}" leakage_curves.py
echo "CURVES DONE $(date)"
