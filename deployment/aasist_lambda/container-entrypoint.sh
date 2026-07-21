#!/bin/sh
set -eu

if [ -z "${AWS_LAMBDA_RUNTIME_API:-}" ]; then
  exec /usr/local/bin/aws-lambda-rie /usr/local/bin/python -m awslambdaric "$@"
fi

exec /usr/local/bin/python -m awslambdaric "$@"

