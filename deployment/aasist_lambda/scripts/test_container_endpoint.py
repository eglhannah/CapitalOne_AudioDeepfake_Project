#!/usr/bin/env python3
"""Check local/container parity and Lambda error responses through RIE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aasist_inference import AudioInferenceService  # noqa: E402
from container_client import function_url_event, invoke, wait_until_ready  # noqa: E402
from inference_contract import MAX_UPLOAD_BYTES  # noqa: E402

EXPECTATIONS = {
    "LA_E_5849185.flac": "bonafide",
    "LA_E_6163791.flac": "spoof",
}


def response_body(response: dict) -> dict:
    return json.loads(response["body"])


def main() -> None:
    wait_until_ready()
    local_service = AudioInferenceService()
    report = {"samples": [], "errors": {}}

    for filename, expected_label in EXPECTATIONS.items():
        path = ROOT / "local_samples" / filename
        if not path.is_file():
            raise SystemExit(f"Missing private parity sample: {path}")
        payload = path.read_bytes()
        local = local_service.score_bytes(payload).inference
        container_response = invoke(function_url_event(payload))
        if container_response["statusCode"] != 200:
            raise AssertionError(container_response)
        container = response_body(container_response)
        if container["classification"] != expected_label:
            raise AssertionError(f"Wrong container label for {filename}")
        if abs(container["spoof_score"] - local.spoof_score) > 1e-5:
            raise AssertionError(f"Local/container score mismatch for {filename}")
        if container["window_count"] != local.window_count:
            raise AssertionError(f"Local/container window mismatch for {filename}")
        report["samples"].append(
            {
                "filename": filename,
                "classification": container["classification"],
                "local_score": local.spoof_score,
                "container_score": container["spoof_score"],
                "absolute_delta": abs(container["spoof_score"] - local.spoof_score),
            }
        )

    error_cases = {
        "method": function_url_event(b"ignored", method="GET"),
        "unsupported": function_url_event(b"not audio"),
        "corrupt": function_url_event(b"fLaC-invalid"),
        "oversized": function_url_event(b"fLaC" + b"\x00" * (MAX_UPLOAD_BYTES - 3)),
    }
    expected_statuses = {"method": 405, "unsupported": 415, "corrupt": 422, "oversized": 413}
    for name, event in error_cases.items():
        response = invoke(event)
        if response["statusCode"] != expected_statuses[name]:
            raise AssertionError(f"Unexpected {name} response: {response}")
        report["errors"][name] = response["statusCode"]

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

