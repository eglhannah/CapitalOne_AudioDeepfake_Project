#!/usr/bin/env python3
"""Invoke the deployed Lambda Function URL with a local audio file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "local_samples" / "LA_E_5849185.flac"
BACKEND_INFO = ROOT / "aws" / "backend-info.json"
DEPLOYMENT_INFO = ROOT / "aws" / "deployment-info.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Lambda Function URL. Defaults to deployment-info.json.")
    parser.add_argument("--passphrase", default=os.environ.get("DEMO_PASSPHRASE"))
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--expect", choices=("bonafide", "spoof"))
    args = parser.parse_args()

    if not args.passphrase:
        raise SystemExit("Provide --passphrase or set DEMO_PASSPHRASE")
    if not args.sample.is_file():
        raise SystemExit(f"Sample file not found: {args.sample}")

    function_url = args.url
    if not function_url:
        info_path = BACKEND_INFO if BACKEND_INFO.is_file() else DEPLOYMENT_INFO
        if not info_path.is_file():
            raise SystemExit(f"Missing backend deployment info: {BACKEND_INFO}")
        function_url = json.loads(info_path.read_text(encoding="utf-8"))["function_url"]

    request = urllib.request.Request(
        function_url,
        data=args.sample.read_bytes(),
        headers={
            "content-type": "audio/flac",
            "x-demo-passcode": args.passphrase,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90.0) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {detail}") from error

    print(json.dumps(body, indent=2, sort_keys=True))
    if args.expect and body.get("classification") != args.expect:
        raise SystemExit(
            f"Expected classification {args.expect}, got {body.get('classification')}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
