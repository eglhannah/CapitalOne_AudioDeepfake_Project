from __future__ import annotations

import base64
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lambda_handler  # noqa: E402
from inference_contract import MAX_UPLOAD_BYTES  # noqa: E402


def event_for(payload: bytes, *, method: str = "POST", headers: dict | None = None) -> dict:
    return {
        "version": "2.0",
        "headers": headers or {},
        "requestContext": {"requestId": "event-request", "http": {"method": method}},
        "body": base64.b64encode(payload).decode("ascii"),
        "isBase64Encoded": True,
    }


class LambdaHandlerTest(unittest.TestCase):
    context = SimpleNamespace(aws_request_id="test-request")

    def test_rejects_non_post_method(self) -> None:
        response = lambda_handler.handler(event_for(b"ignored", method="GET"), self.context)
        self.assertEqual(response["statusCode"], 405)
        self.assertEqual(json.loads(response["body"])["error"], "method_not_allowed")
        self.assertIn("access-control-allow-origin", response["headers"])

    def test_allows_cors_preflight_without_passcode(self) -> None:
        with patch.dict(os.environ, {"DEMO_PASSPHRASE": "secret"}, clear=False):
            response = lambda_handler.handler(
                event_for(b"", method="OPTIONS", headers={"origin": "http://localhost:8765"}),
                self.context,
            )
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(response["headers"]["access-control-allow-methods"], "POST,OPTIONS")
        self.assertIn("x-demo-passcode", response["headers"]["access-control-allow-headers"])

    def test_uses_configured_cors_origin(self) -> None:
        with patch.dict(os.environ, {"DEMO_ALLOWED_ORIGIN": "http://localhost:8765"}, clear=False):
            response = lambda_handler.handler(event_for(b"not audio"), self.context)
        self.assertEqual(
            response["headers"]["access-control-allow-origin"],
            "http://localhost:8765",
        )

    def test_can_echo_request_origin_when_configured(self) -> None:
        with patch.dict(os.environ, {"DEMO_ALLOWED_ORIGIN": "request"}, clear=False):
            response = lambda_handler.handler(
                event_for(b"not audio", headers={"Origin": "https://demo.example"}),
                self.context,
            )
        self.assertEqual(response["headers"]["access-control-allow-origin"], "https://demo.example")

    def test_rejects_wrong_demo_passcode_before_decoding(self) -> None:
        with patch.dict(os.environ, {"DEMO_PASSPHRASE": "secret"}, clear=False):
            missing = lambda_handler.handler(event_for(b"not audio"), self.context)
            wrong = lambda_handler.handler(
                event_for(b"not audio", headers={"x-demo-passcode": "incorrect"}),
                self.context,
            )
        self.assertEqual(missing["statusCode"], 401)
        self.assertEqual(wrong["statusCode"], 401)
        self.assertEqual(json.loads(missing["body"])["error"], "unauthorized")

    def test_accepts_correct_demo_passcode(self) -> None:
        with patch.dict(os.environ, {"DEMO_PASSPHRASE": "secret"}, clear=False):
            response = lambda_handler.handler(
                event_for(b"not audio", headers={"X-Demo-Passcode": "secret"}),
                self.context,
            )
        self.assertEqual(response["statusCode"], 415)

    def test_rejects_missing_and_invalid_base64_body(self) -> None:
        missing = lambda_handler.handler(
            {"requestContext": {"http": {"method": "POST"}}}, self.context
        )
        self.assertEqual(missing["statusCode"], 400)
        invalid = lambda_handler.handler(
            {
                "requestContext": {"http": {"method": "POST"}},
                "body": "%%%",
                "isBase64Encoded": True,
            },
            self.context,
        )
        self.assertEqual(invalid["statusCode"], 400)

    def test_maps_unsupported_corrupt_and_oversized_audio_errors(self) -> None:
        unsupported = lambda_handler.handler(event_for(b"not audio"), self.context)
        self.assertEqual(unsupported["statusCode"], 415)
        corrupt = lambda_handler.handler(event_for(b"fLaC-invalid"), self.context)
        self.assertEqual(corrupt["statusCode"], 422)
        oversized = lambda_handler.handler(
            event_for(b"fLaC" + b"\x00" * (MAX_UPLOAD_BYTES - 3)), self.context
        )
        self.assertEqual(oversized["statusCode"], 413)


class PrivateSampleLambdaIntegrationTest(unittest.TestCase):
    context = SimpleNamespace(aws_request_id="private-sample-test")
    expectations = {
        "LA_E_5849185.flac": "bonafide",
        "LA_E_6163791.flac": "spoof",
    }

    def test_known_samples_match_confirmed_labels(self) -> None:
        for filename, expected_label in self.expectations.items():
            path = ROOT / "local_samples" / filename
            if not path.is_file():
                self.skipTest("Private ASVspoof samples are not available")
            with self.subTest(filename=filename):
                response = lambda_handler.handler(event_for(path.read_bytes()), self.context)
                self.assertEqual(response["statusCode"], 200)
                body = json.loads(response["body"])
                self.assertEqual(body["classification"], expected_label)
                self.assertEqual(body["request_id"], "private-sample-test")
                self.assertNotIn("windows", body)


if __name__ == "__main__":
    unittest.main()
