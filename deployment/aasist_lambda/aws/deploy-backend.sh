#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_command aws
require_command docker
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at PYTHON_BIN=${PYTHON_BIN}" >&2
  exit 1
fi
require_expected_account

if [[ -z "${DEMO_PASSPHRASE:-}" ]]; then
  echo "Set DEMO_PASSPHRASE before deploying the backend." >&2
  exit 1
fi

ACCOUNT_ID="$(current_account_id)"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REMOTE_IMAGE_TAG="${REMOTE_IMAGE_TAG:-phase5.5}"
REMOTE_IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY_NAME}:${REMOTE_IMAGE_TAG}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
BACKEND_ALLOWED_ORIGIN="${BACKEND_ALLOWED_ORIGIN:-*}"

echo "Deploying backend ${PROJECT_NAME} to account ${ACCOUNT_ID} in ${AWS_REGION}"

if ! aws_cli ecr describe-repositories \
  --repository-names "$ECR_REPOSITORY_NAME" >/dev/null 2>&1; then
  aws_cli ecr create-repository \
    --repository-name "$ECR_REPOSITORY_NAME" \
    --image-scanning-configuration scanOnPush=true \
    --tags \
      Key=Project,Value="$PROJECT_NAME" \
      Key=Purpose,Value=audio-deepfake-demo \
      Key=ManagedBy,Value=phase6-scripts >/dev/null
fi

aws_cli ecr get-login-password \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null

(cd "$ROOT_DIR" && docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --load \
  --tag "$IMAGE_NAME" \
  .)

docker tag "$IMAGE_NAME" "$REMOTE_IMAGE_URI"
docker push "$REMOTE_IMAGE_URI"

TRUST_POLICY_FILE="$(mktemp)"
cat > "$TRUST_POLICY_FILE" <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON

if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://${TRUST_POLICY_FILE}" \
    --tags \
      Key=Project,Value="$PROJECT_NAME" \
      Key=Purpose,Value=audio-deepfake-demo \
      Key=ManagedBy,Value=phase6-scripts >/dev/null
else
  aws iam update-assume-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-document "file://${TRUST_POLICY_FILE}" >/dev/null
fi
rm -f "$TRUST_POLICY_FILE"

aws iam attach-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null || true

sleep 10

ENV_FILE="$(mktemp)"
DEMO_PASSPHRASE="$DEMO_PASSPHRASE" BACKEND_ALLOWED_ORIGIN="$BACKEND_ALLOWED_ORIGIN" "$PYTHON_BIN" - <<'PY' > "$ENV_FILE"
import json
import os

print(json.dumps({
    "Variables": {
        "DEMO_PASSPHRASE": os.environ["DEMO_PASSPHRASE"],
        "DEMO_ALLOWED_ORIGIN": os.environ["BACKEND_ALLOWED_ORIGIN"],
    }
}))
PY

if ! aws_cli lambda get-function --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  aws_cli lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --package-type Image \
    --code ImageUri="$REMOTE_IMAGE_URI" \
    --role "$ROLE_ARN" \
    --architectures x86_64 \
    --memory-size "$LAMBDA_MEMORY_MB" \
    --timeout "$LAMBDA_TIMEOUT_SECONDS" \
    --environment "file://${ENV_FILE}" \
    --tags Project="$PROJECT_NAME",Purpose=audio-deepfake-demo,ManagedBy=phase6-scripts >/dev/null
  aws_cli lambda wait function-active --function-name "$FUNCTION_NAME"
else
  aws_cli lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --image-uri "$REMOTE_IMAGE_URI" >/dev/null
  aws_cli lambda wait function-updated --function-name "$FUNCTION_NAME"
  aws_cli lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --memory-size "$LAMBDA_MEMORY_MB" \
    --timeout "$LAMBDA_TIMEOUT_SECONDS" \
    --environment "file://${ENV_FILE}" >/dev/null
  aws_cli lambda wait function-updated --function-name "$FUNCTION_NAME"
fi
rm -f "$ENV_FILE"

if [[ -n "$LAMBDA_RESERVED_CONCURRENCY" ]]; then
  aws_cli lambda put-function-concurrency \
    --function-name "$FUNCTION_NAME" \
    --reserved-concurrent-executions "$LAMBDA_RESERVED_CONCURRENCY" >/dev/null
else
  aws_cli lambda delete-function-concurrency \
    --function-name "$FUNCTION_NAME" >/dev/null 2>&1 || true
fi

LOG_GROUP="/aws/lambda/${FUNCTION_NAME}"
aws_cli logs create-log-group --log-group-name "$LOG_GROUP" >/dev/null 2>&1 || true
aws_cli logs put-retention-policy \
  --log-group-name "$LOG_GROUP" \
  --retention-in-days "$LOG_RETENTION_DAYS" >/dev/null

CORS_FILE="$(mktemp)"
BACKEND_ALLOWED_ORIGIN="$BACKEND_ALLOWED_ORIGIN" "$PYTHON_BIN" - <<'PY' > "$CORS_FILE"
import json
import os

print(json.dumps({
    "AllowOrigins": [os.environ["BACKEND_ALLOWED_ORIGIN"]],
    "AllowMethods": ["POST"],
    "AllowHeaders": ["content-type", "x-demo-passcode"],
    "MaxAge": 3600,
}))
PY

if ! aws_cli lambda get-function-url-config --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  FUNCTION_URL="$(aws_cli lambda create-function-url-config \
    --function-name "$FUNCTION_NAME" \
    --auth-type NONE \
    --cors "file://${CORS_FILE}" \
    --query FunctionUrl \
    --output text)"
else
  FUNCTION_URL="$(aws_cli lambda update-function-url-config \
    --function-name "$FUNCTION_NAME" \
    --auth-type NONE \
    --cors "file://${CORS_FILE}" \
    --query FunctionUrl \
    --output text)"
fi
rm -f "$CORS_FILE"

aws_cli lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE >/dev/null 2>&1 || true

aws_cli lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id FunctionURLInvokeAllowPublicAccess \
  --action lambda:InvokeFunction \
  --principal "*" \
  --invoked-via-function-url >/dev/null 2>&1 || true

cat > "$BACKEND_INFO_FILE" <<JSON
{
  "account_id": "${ACCOUNT_ID}",
  "region": "${AWS_REGION}",
  "project_name": "${PROJECT_NAME}",
  "function_name": "${FUNCTION_NAME}",
  "function_url": "${FUNCTION_URL}",
  "ecr_repository": "${ECR_REPOSITORY_NAME}",
  "image_uri": "${REMOTE_IMAGE_URI}",
  "lambda_memory_mb": ${LAMBDA_MEMORY_MB},
  "lambda_timeout_seconds": ${LAMBDA_TIMEOUT_SECONDS},
  "lambda_reserved_concurrency": ${LAMBDA_RESERVED_CONCURRENCY:-null}
}
JSON

cp "$BACKEND_INFO_FILE" "$LEGACY_DEPLOYMENT_INFO_FILE"

echo
echo "Backend deployment complete."
echo "Function URL: ${FUNCTION_URL}"
echo "Backend info: ${BACKEND_INFO_FILE}"
