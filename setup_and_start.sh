#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 is required. Select a Python 3.12 GPU template or set PYTHON_BIN to its interpreter." >&2
  exit 1
fi
exec "$PYTHON_BIN" "$REPO_DIR/scripts/setup_v1.py" "$@"
