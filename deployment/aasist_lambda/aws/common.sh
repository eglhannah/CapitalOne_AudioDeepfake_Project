#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-aasist-audio-deepfake-demo}"
IMAGE_NAME="${IMAGE_NAME:-aasist-lambda:phase5.5}"
LAMBDA_MEMORY_MB="${LAMBDA_MEMORY_MB:-2048}"
LAMBDA_TIMEOUT_SECONDS="${LAMBDA_TIMEOUT_SECONDS:-90}"
LAMBDA_RESERVED_CONCURRENCY="${LAMBDA_RESERVED_CONCURRENCY:-}"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-7}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CLOUDFRONT_PRICE_CLASS="${CLOUDFRONT_PRICE_CLASS:-PriceClass_100}"
WAIT_FOR_CLOUDFRONT="${WAIT_FOR_CLOUDFRONT:-true}"

FUNCTION_NAME="${FUNCTION_NAME:-$PROJECT_NAME}"
ECR_REPOSITORY_NAME="${ECR_REPOSITORY_NAME:-$PROJECT_NAME}"
ROLE_NAME="${ROLE_NAME:-$PROJECT_NAME-role}"
BUCKET_NAME="${BUCKET_NAME:-$PROJECT_NAME-${EXPECTED_AWS_ACCOUNT_ID:-unknown}-$AWS_REGION}"
CLOUDFRONT_COMMENT="${CLOUDFRONT_COMMENT:-$PROJECT_NAME HTTPS static frontend}"
BACKEND_INFO_FILE="$ROOT_DIR/aws/backend-info.json"
FRONTEND_INFO_FILE="$ROOT_DIR/aws/frontend-info.json"
LEGACY_DEPLOYMENT_INFO_FILE="$ROOT_DIR/aws/deployment-info.json"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

aws_cli() {
  aws --region "$AWS_REGION" "$@"
}

current_account_id() {
  aws_cli sts get-caller-identity --query Account --output text
}

require_expected_account() {
  if [[ -z "${EXPECTED_AWS_ACCOUNT_ID:-}" ]]; then
    echo "Set EXPECTED_AWS_ACCOUNT_ID before running this script." >&2
    echo "Current account appears to be: $(current_account_id)" >&2
    exit 1
  fi

  local actual_account_id
  actual_account_id="$(current_account_id)"
  if [[ "$actual_account_id" != "$EXPECTED_AWS_ACCOUNT_ID" ]]; then
    echo "Refusing to continue: expected AWS account $EXPECTED_AWS_ACCOUNT_ID, got $actual_account_id." >&2
    exit 1
  fi
}

website_endpoint() {
  echo "http://${BUCKET_NAME}.s3-website-${AWS_REGION}.amazonaws.com"
}

bucket_regional_domain_name() {
  echo "${BUCKET_NAME}.s3.${AWS_REGION}.amazonaws.com"
}

tag_args() {
  printf '%s\n' \
    Key=Project,Value="$PROJECT_NAME" \
    Key=Purpose,Value=audio-deepfake-demo \
    Key=ManagedBy,Value=phase6-scripts
}
