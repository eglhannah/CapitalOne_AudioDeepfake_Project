#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_command aws
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at PYTHON_BIN=${PYTHON_BIN}" >&2
  exit 1
fi

DEPLOYMENT_INFO_FILE="$ROOT_DIR/aws/deployment-info.json"
if [[ ! -f "$DEPLOYMENT_INFO_FILE" ]]; then
  echo "Missing deployment info: $DEPLOYMENT_INFO_FILE" >&2
  echo "Run ./aws/deploy.sh first, or rerun deploy to recreate this file." >&2
  exit 1
fi

read -r DEPLOYED_ACCOUNT_ID DEPLOYED_REGION DEPLOYED_BUCKET FUNCTION_URL < <(
  DEPLOYMENT_INFO_FILE="$DEPLOYMENT_INFO_FILE" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

info = json.loads(Path(os.environ["DEPLOYMENT_INFO_FILE"]).read_text(encoding="utf-8"))
print(info["account_id"], info["region"], info["s3_bucket"], info["function_url"])
PY
)

ACTUAL_ACCOUNT_ID="$(aws --region "$DEPLOYED_REGION" sts get-caller-identity --query Account --output text)"
if [[ "$ACTUAL_ACCOUNT_ID" != "$DEPLOYED_ACCOUNT_ID" ]]; then
  echo "Refusing to update static client: expected AWS account $DEPLOYED_ACCOUNT_ID, got $ACTUAL_ACCOUNT_ID." >&2
  exit 1
fi

aws --region "$DEPLOYED_REGION" s3 sync "$ROOT_DIR/demo_client" "s3://${DEPLOYED_BUCKET}/" \
  --delete \
  --exclude "config.js" >/dev/null

CONFIG_FILE="$(mktemp)"
FUNCTION_URL="$FUNCTION_URL" "$PYTHON_BIN" - <<'PY' > "$CONFIG_FILE"
import json
import os

print("window.AASIST_DEMO_CONFIG = " + json.dumps({
    "endpoint": os.environ["FUNCTION_URL"],
}, indent=2) + ";")
PY

aws --region "$DEPLOYED_REGION" s3 cp "$CONFIG_FILE" "s3://${DEPLOYED_BUCKET}/config.js" \
  --content-type "application/javascript" \
  --cache-control "no-store" >/dev/null
rm -f "$CONFIG_FILE"

echo "Updated static demo client in s3://${DEPLOYED_BUCKET}/"
