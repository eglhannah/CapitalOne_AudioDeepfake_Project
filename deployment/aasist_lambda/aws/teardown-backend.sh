#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_command aws
require_expected_account

if [[ "${CONFIRM_TEARDOWN:-}" != "$PROJECT_NAME" ]]; then
  echo "Refusing to tear down backend without explicit confirmation." >&2
  echo "Run with: CONFIRM_TEARDOWN=${PROJECT_NAME} EXPECTED_AWS_ACCOUNT_ID=${EXPECTED_AWS_ACCOUNT_ID} $0" >&2
  exit 1
fi

ACCOUNT_ID="$(current_account_id)"
echo "Tearing down backend ${PROJECT_NAME} from account ${ACCOUNT_ID} in ${AWS_REGION}"

if aws_cli lambda get-function-url-config --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  aws_cli lambda delete-function-url-config --function-name "$FUNCTION_NAME" >/dev/null
fi

aws_cli lambda remove-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id FunctionURLAllowPublicAccess >/dev/null 2>&1 || true

aws_cli lambda remove-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id FunctionURLInvokeAllowPublicAccess >/dev/null 2>&1 || true

if aws_cli lambda get-function --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  aws_cli lambda delete-function --function-name "$FUNCTION_NAME" >/dev/null
fi

aws_cli logs delete-log-group \
  --log-group-name "/aws/lambda/${FUNCTION_NAME}" >/dev/null 2>&1 || true

if aws_cli ecr describe-repositories \
  --repository-names "$ECR_REPOSITORY_NAME" >/dev/null 2>&1; then
  aws_cli ecr delete-repository \
    --repository-name "$ECR_REPOSITORY_NAME" \
    --force >/dev/null
fi

aws iam detach-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null 2>&1 || true

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam delete-role --role-name "$ROLE_NAME" >/dev/null
fi

rm -f "$BACKEND_INFO_FILE"

if [[ -f "$FRONTEND_INFO_FILE" ]]; then
  cp "$FRONTEND_INFO_FILE" "$LEGACY_DEPLOYMENT_INFO_FILE"
else
  rm -f "$LEGACY_DEPLOYMENT_INFO_FILE"
fi

echo "Backend teardown complete."
