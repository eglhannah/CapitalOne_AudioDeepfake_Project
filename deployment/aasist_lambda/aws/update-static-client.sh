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

if [[ ! -f "$BACKEND_INFO_FILE" || ! -f "$FRONTEND_INFO_FILE" ]]; then
  echo "Missing backend/frontend deployment info." >&2
  echo "Run ./aws/deploy-backend.sh and ./aws/deploy-frontend.sh first." >&2
  exit 1
fi

read -r DEPLOYED_ACCOUNT_ID DEPLOYED_REGION DEPLOYED_BUCKET DISTRIBUTION_ID FUNCTION_URL < <(
  BACKEND_INFO_FILE="$BACKEND_INFO_FILE" FRONTEND_INFO_FILE="$FRONTEND_INFO_FILE" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

backend = json.loads(Path(os.environ["BACKEND_INFO_FILE"]).read_text(encoding="utf-8"))
frontend = json.loads(Path(os.environ["FRONTEND_INFO_FILE"]).read_text(encoding="utf-8"))
print(
    frontend["account_id"],
    frontend["region"],
    frontend["s3_bucket"],
    frontend["cloudfront_distribution_id"],
    backend["function_url"],
)
PY
)

ACTUAL_ACCOUNT_ID="$(aws --region "$DEPLOYED_REGION" sts get-caller-identity --query Account --output text)"
if [[ "$ACTUAL_ACCOUNT_ID" != "$DEPLOYED_ACCOUNT_ID" ]]; then
  echo "Refusing to update static client: expected AWS account $DEPLOYED_ACCOUNT_ID, got $ACTUAL_ACCOUNT_ID." >&2
  exit 1
fi

aws --region "$DEPLOYED_REGION" s3 sync "$ROOT_DIR/demo_client" "s3://${DEPLOYED_BUCKET}/" \
  --delete \
  --exclude "config.js" \
  --cache-control "no-cache" >/dev/null

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

CALLER_REFERENCE="${PROJECT_NAME}-static-refresh-$(date +%s)"
INVALIDATION_BATCH="$(mktemp)"
CALLER_REFERENCE="$CALLER_REFERENCE" "$PYTHON_BIN" - <<'PY' > "$INVALIDATION_BATCH"
import json
import os

print(json.dumps({
    "Paths": {
        "Quantity": 2,
        "Items": ["/index.html", "/config.js"],
    },
    "CallerReference": os.environ["CALLER_REFERENCE"],
}))
PY
aws cloudfront create-invalidation \
  --distribution-id "$DISTRIBUTION_ID" \
  --invalidation-batch "file://${INVALIDATION_BATCH}" >/dev/null
rm -f "$INVALIDATION_BATCH"

echo "Updated static demo client in s3://${DEPLOYED_BUCKET}/"
