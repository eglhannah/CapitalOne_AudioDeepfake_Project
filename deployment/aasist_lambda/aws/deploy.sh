#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "deploy.sh is a compatibility wrapper for deploy-backend.sh."
exec "$SCRIPT_DIR/deploy-backend.sh" "$@"
