#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at PYTHON_BIN=${PYTHON_BIN}" >&2
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/smoke_test.py" "$@"
