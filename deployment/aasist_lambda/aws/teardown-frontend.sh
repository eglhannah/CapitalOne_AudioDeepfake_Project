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
require_expected_account

if [[ "${CONFIRM_TEARDOWN:-}" != "$PROJECT_NAME" ]]; then
  echo "Refusing to tear down frontend without explicit confirmation." >&2
  echo "Run with: CONFIRM_TEARDOWN=${PROJECT_NAME} EXPECTED_AWS_ACCOUNT_ID=${EXPECTED_AWS_ACCOUNT_ID} $0" >&2
  exit 1
fi

ACCOUNT_ID="$(current_account_id)"
echo "Tearing down HTTPS frontend ${PROJECT_NAME} from account ${ACCOUNT_ID} in ${AWS_REGION}"

DISTRIBUTION_ID=""
if [[ -f "$FRONTEND_INFO_FILE" ]]; then
  read -r INFO_ACCOUNT_ID INFO_REGION DISTRIBUTION_ID < <(
    FRONTEND_INFO_FILE="$FRONTEND_INFO_FILE" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

info = json.loads(Path(os.environ["FRONTEND_INFO_FILE"]).read_text(encoding="utf-8"))
print(info["account_id"], info["region"], info.get("cloudfront_distribution_id", ""))
PY
  )
  if [[ "$INFO_ACCOUNT_ID" != "$ACCOUNT_ID" || "$INFO_REGION" != "$AWS_REGION" ]]; then
    echo "Refusing to use frontend info for ${INFO_ACCOUNT_ID}/${INFO_REGION} from current ${ACCOUNT_ID}/${AWS_REGION}." >&2
    exit 1
  fi
fi

if [[ -n "$DISTRIBUTION_ID" ]]; then
  if DISTRIBUTION_OUTPUT="$(aws cloudfront get-distribution-config \
    --id "$DISTRIBUTION_ID" \
    --output json 2>/dev/null)"; then
    ETAG="$(DISTRIBUTION_OUTPUT="$DISTRIBUTION_OUTPUT" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["DISTRIBUTION_OUTPUT"])
print(data["ETag"])
PY
)"
    ENABLED="$(DISTRIBUTION_OUTPUT="$DISTRIBUTION_OUTPUT" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["DISTRIBUTION_OUTPUT"])
print("true" if data["DistributionConfig"].get("Enabled") else "false")
PY
)"
    if [[ "$ENABLED" == "true" ]]; then
      DISABLED_CONFIG="$(mktemp)"
      DISTRIBUTION_OUTPUT="$DISTRIBUTION_OUTPUT" "$PYTHON_BIN" - <<'PY' > "$DISABLED_CONFIG"
import json
import os

data = json.loads(os.environ["DISTRIBUTION_OUTPUT"])
config = data["DistributionConfig"]
config["Enabled"] = False
print(json.dumps(config))
PY
      aws cloudfront update-distribution \
        --id "$DISTRIBUTION_ID" \
        --if-match "$ETAG" \
        --distribution-config "file://${DISABLED_CONFIG}" >/dev/null
      rm -f "$DISABLED_CONFIG"
      echo "Waiting for CloudFront distribution ${DISTRIBUTION_ID} to disable..."
      aws cloudfront wait distribution-deployed --id "$DISTRIBUTION_ID"
      DISTRIBUTION_OUTPUT="$(aws cloudfront get-distribution-config \
        --id "$DISTRIBUTION_ID" \
        --output json)"
      ETAG="$(DISTRIBUTION_OUTPUT="$DISTRIBUTION_OUTPUT" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["DISTRIBUTION_OUTPUT"])
print(data["ETag"])
PY
)"
    fi
    aws cloudfront delete-distribution \
      --id "$DISTRIBUTION_ID" \
      --if-match "$ETAG" >/dev/null
  else
    echo "CloudFront distribution ${DISTRIBUTION_ID} was not found; continuing." >&2
  fi
fi

if aws_cli s3api head-bucket --bucket "$BUCKET_NAME" >/dev/null 2>&1; then
  aws_cli s3 rm "s3://${BUCKET_NAME}" --recursive >/dev/null
  aws_cli s3api delete-bucket-policy --bucket "$BUCKET_NAME" >/dev/null 2>&1 || true
  aws_cli s3api delete-public-access-block --bucket "$BUCKET_NAME" >/dev/null 2>&1 || true
  aws_cli s3api delete-bucket-website --bucket "$BUCKET_NAME" >/dev/null 2>&1 || true
  aws_cli s3api delete-bucket --bucket "$BUCKET_NAME" >/dev/null
fi

rm -f "$FRONTEND_INFO_FILE"

if [[ -f "$BACKEND_INFO_FILE" ]]; then
  cp "$BACKEND_INFO_FILE" "$LEGACY_DEPLOYMENT_INFO_FILE"
else
  rm -f "$LEGACY_DEPLOYMENT_INFO_FILE"
fi

echo "Frontend teardown complete."
