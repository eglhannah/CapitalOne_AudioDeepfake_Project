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

if [[ ! -f "$BACKEND_INFO_FILE" ]]; then
  echo "Missing backend info: $BACKEND_INFO_FILE" >&2
  echo "Run ./aws/deploy-backend.sh first." >&2
  exit 1
fi

ACCOUNT_ID="$(current_account_id)"
read -r BACKEND_ACCOUNT_ID BACKEND_REGION FUNCTION_URL < <(
  BACKEND_INFO_FILE="$BACKEND_INFO_FILE" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

info = json.loads(Path(os.environ["BACKEND_INFO_FILE"]).read_text(encoding="utf-8"))
print(info["account_id"], info["region"], info["function_url"])
PY
)

if [[ "$BACKEND_ACCOUNT_ID" != "$ACCOUNT_ID" || "$BACKEND_REGION" != "$AWS_REGION" ]]; then
  echo "Backend info does not match current account/region." >&2
  echo "backend=${BACKEND_ACCOUNT_ID}/${BACKEND_REGION}, current=${ACCOUNT_ID}/${AWS_REGION}" >&2
  exit 1
fi

echo "Deploying HTTPS frontend ${PROJECT_NAME} to account ${ACCOUNT_ID} in ${AWS_REGION}"

if ! aws_cli s3api head-bucket --bucket "$BUCKET_NAME" >/dev/null 2>&1; then
  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws_cli s3api create-bucket --bucket "$BUCKET_NAME" >/dev/null
  else
    aws_cli s3api create-bucket \
      --bucket "$BUCKET_NAME" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null
  fi
fi

aws_cli s3api put-public-access-block \
  --bucket "$BUCKET_NAME" \
  --public-access-block-configuration \
  BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false >/dev/null

BUCKET_POLICY_FILE="$(mktemp)"
cat > "$BUCKET_POLICY_FILE" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadDemoClient",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::${BUCKET_NAME}/*"
    }
  ]
}
JSON
aws_cli s3api put-bucket-policy \
  --bucket "$BUCKET_NAME" \
  --policy "file://${BUCKET_POLICY_FILE}" >/dev/null
rm -f "$BUCKET_POLICY_FILE"

aws_cli s3 sync "$ROOT_DIR/demo_client" "s3://${BUCKET_NAME}/" \
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
aws_cli s3 cp "$CONFIG_FILE" "s3://${BUCKET_NAME}/config.js" \
  --content-type "application/javascript" \
  --cache-control "no-store" >/dev/null
rm -f "$CONFIG_FILE"

DISTRIBUTION_ID=""
DISTRIBUTION_DOMAIN=""
ETAG=""
if [[ -f "$FRONTEND_INFO_FILE" ]]; then
  read -r INFO_ACCOUNT_ID INFO_REGION DISTRIBUTION_ID DISTRIBUTION_DOMAIN < <(
    FRONTEND_INFO_FILE="$FRONTEND_INFO_FILE" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

info = json.loads(Path(os.environ["FRONTEND_INFO_FILE"]).read_text(encoding="utf-8"))
print(info["account_id"], info["region"], info.get("cloudfront_distribution_id", ""), info.get("cloudfront_domain", ""))
PY
  )
  if [[ "$INFO_ACCOUNT_ID" != "$ACCOUNT_ID" || "$INFO_REGION" != "$AWS_REGION" ]]; then
    echo "Ignoring stale frontend info for ${INFO_ACCOUNT_ID}/${INFO_REGION}." >&2
    DISTRIBUTION_ID=""
    DISTRIBUTION_DOMAIN=""
  fi
fi

if [[ -n "$DISTRIBUTION_ID" ]]; then
  if ! DISTRIBUTION_JSON="$(aws cloudfront get-distribution \
    --id "$DISTRIBUTION_ID" \
    --output json 2>/dev/null)"; then
    echo "Stored CloudFront distribution was not found; creating a new one." >&2
    DISTRIBUTION_ID=""
    DISTRIBUTION_DOMAIN=""
  else
    DISTRIBUTION_DOMAIN="$(DISTRIBUTION_JSON="$DISTRIBUTION_JSON" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["DISTRIBUTION_JSON"])
print(data["Distribution"]["DomainName"])
PY
)"
  fi
fi

if [[ -z "$DISTRIBUTION_ID" ]]; then
  CALLER_REFERENCE="${PROJECT_NAME}-$(date +%s)"
  ORIGIN_DOMAIN="$(bucket_regional_domain_name)"
  DISTRIBUTION_CONFIG="$(mktemp)"
  CALLER_REFERENCE="$CALLER_REFERENCE" ORIGIN_DOMAIN="$ORIGIN_DOMAIN" CLOUDFRONT_COMMENT="$CLOUDFRONT_COMMENT" CLOUDFRONT_PRICE_CLASS="$CLOUDFRONT_PRICE_CLASS" "$PYTHON_BIN" - <<'PY' > "$DISTRIBUTION_CONFIG"
import json
import os

origin_id = "s3-static-frontend"
print(json.dumps({
    "CallerReference": os.environ["CALLER_REFERENCE"],
    "Comment": os.environ["CLOUDFRONT_COMMENT"],
    "Enabled": True,
    "DefaultRootObject": "index.html",
    "PriceClass": os.environ["CLOUDFRONT_PRICE_CLASS"],
    "Origins": {
        "Quantity": 1,
        "Items": [{
            "Id": origin_id,
            "DomainName": os.environ["ORIGIN_DOMAIN"],
            "S3OriginConfig": {
                "OriginAccessIdentity": ""
            },
        }],
    },
    "DefaultCacheBehavior": {
        "TargetOriginId": origin_id,
        "ViewerProtocolPolicy": "redirect-to-https",
        "AllowedMethods": {
            "Quantity": 3,
            "Items": ["GET", "HEAD", "OPTIONS"],
            "CachedMethods": {
                "Quantity": 2,
                "Items": ["GET", "HEAD"],
            },
        },
        "ForwardedValues": {
            "QueryString": False,
            "Cookies": {"Forward": "none"},
        },
        "MinTTL": 0,
        "DefaultTTL": 60,
        "MaxTTL": 300,
        "Compress": True,
    },
    "CustomErrorResponses": {
        "Quantity": 1,
        "Items": [{
            "ErrorCode": 404,
            "ResponsePagePath": "/index.html",
            "ResponseCode": "200",
            "ErrorCachingMinTTL": 0,
        }],
    },
}))
PY
  CREATE_OUTPUT="$(aws cloudfront create-distribution \
    --distribution-config "file://${DISTRIBUTION_CONFIG}" \
    --output json)"
  rm -f "$DISTRIBUTION_CONFIG"
  DISTRIBUTION_ID="$(CREATE_OUTPUT="$CREATE_OUTPUT" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["CREATE_OUTPUT"])
print(data["Distribution"]["Id"])
PY
)"
  DISTRIBUTION_DOMAIN="$(CREATE_OUTPUT="$CREATE_OUTPUT" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["CREATE_OUTPUT"])
print(data["Distribution"]["DomainName"])
PY
)"
  if [[ "$WAIT_FOR_CLOUDFRONT" == "true" ]]; then
    echo "Waiting for CloudFront distribution ${DISTRIBUTION_ID} to deploy..."
    aws cloudfront wait distribution-deployed --id "$DISTRIBUTION_ID"
  else
    echo "CloudFront distribution ${DISTRIBUTION_ID} is being created asynchronously."
  fi
else
  INVALIDATION_BATCH="$(mktemp)"
  CALLER_REFERENCE="${PROJECT_NAME}-static-refresh-$(date +%s)"
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
fi

FRONTEND_URL="https://${DISTRIBUTION_DOMAIN}"

CORS_FILE="$(mktemp)"
FRONTEND_URL="$FRONTEND_URL" "$PYTHON_BIN" - <<'PY' > "$CORS_FILE"
import json
import os

print(json.dumps({
    "AllowOrigins": [os.environ["FRONTEND_URL"]],
    "AllowMethods": ["POST"],
    "AllowHeaders": ["content-type", "x-demo-passcode"],
    "MaxAge": 3600,
}))
PY
aws_cli lambda update-function-url-config \
  --function-name "$FUNCTION_NAME" \
  --auth-type NONE \
  --cors "file://${CORS_FILE}" >/dev/null
rm -f "$CORS_FILE"

cat > "$FRONTEND_INFO_FILE" <<JSON
{
  "account_id": "${ACCOUNT_ID}",
  "region": "${AWS_REGION}",
  "project_name": "${PROJECT_NAME}",
  "s3_bucket": "${BUCKET_NAME}",
  "cloudfront_distribution_id": "${DISTRIBUTION_ID}",
  "cloudfront_domain": "${DISTRIBUTION_DOMAIN}",
  "frontend_url": "${FRONTEND_URL}",
  "function_name": "${FUNCTION_NAME}",
  "function_url": "${FUNCTION_URL}"
}
JSON

BACKEND_INFO_FILE="$BACKEND_INFO_FILE" FRONTEND_INFO_FILE="$FRONTEND_INFO_FILE" LEGACY_DEPLOYMENT_INFO_FILE="$LEGACY_DEPLOYMENT_INFO_FILE" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

backend_path = Path(os.environ["BACKEND_INFO_FILE"])
frontend_path = Path(os.environ["FRONTEND_INFO_FILE"])
legacy_path = Path(os.environ["LEGACY_DEPLOYMENT_INFO_FILE"])
merged = {}
if backend_path.exists():
    merged.update(json.loads(backend_path.read_text(encoding="utf-8")))
if frontend_path.exists():
    merged.update(json.loads(frontend_path.read_text(encoding="utf-8")))
legacy_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "Frontend deployment complete."
echo "Frontend URL: ${FRONTEND_URL}"
echo "S3 bucket: ${BUCKET_NAME}"
echo "CloudFront distribution: ${DISTRIBUTION_ID}"
echo "Frontend info: ${FRONTEND_INFO_FILE}"
