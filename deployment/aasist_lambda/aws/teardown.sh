#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "teardown.sh is a compatibility wrapper for teardown-backend.sh."
exec "$SCRIPT_DIR/teardown-backend.sh" "$@"
