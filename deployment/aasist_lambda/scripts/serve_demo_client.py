"""Serve the demo UI and proxy browser uploads to the local Lambda emulator.

The AWS Lambda Runtime Interface Emulator accepts Lambda invocation JSON rather
than normal browser file uploads. This development-only server bridges that gap:
the browser POSTs raw audio to /infer, and this script invokes the local
container with a Function URL-shaped event.
"""

from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEMO_CLIENT_DIR = ROOT / "demo_client"
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 8765
DEFAULT_LAMBDA_URL = "http://127.0.0.1:9000/2015-03-31/functions/function/invocations"
MAX_PROXY_UPLOAD_BYTES = 4 * 1024 * 1024


def _event_for(payload: bytes, headers: dict[str, str]) -> dict[str, Any]:
    return {
        "version": "2.0",
        "headers": headers,
        "requestContext": {
            "requestId": "local-demo-request",
            "http": {"method": "POST"},
        },
        "body": base64.b64encode(payload).decode("ascii"),
        "isBase64Encoded": True,
    }


class DemoRequestHandler(SimpleHTTPRequestHandler):
    lambda_url = DEFAULT_LAMBDA_URL

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(DEMO_CLIENT_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("cache-control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path != "/infer":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "POST,OPTIONS")
        self.send_header("access-control-allow-headers", "content-type,x-demo-passcode")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path != "/infer":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_length = self.headers.get("content-length")
        if content_length is None:
            self._send_json(
                HTTPStatus.LENGTH_REQUIRED,
                {"error": "length_required", "message": "Missing content-length header"},
            )
            return

        try:
            byte_count = int(content_length)
        except ValueError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "message": "Invalid content-length header"},
            )
            return

        if byte_count > MAX_PROXY_UPLOAD_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "upload_too_large", "message": "Upload exceeds 4 MiB"},
            )
            return

        payload = self.rfile.read(byte_count)
        event_headers = {
            "content-type": self.headers.get("content-type", "application/octet-stream"),
            "origin": f"http://{self.headers.get('host', f'{DEFAULT_LISTEN_HOST}:{DEFAULT_LISTEN_PORT}')}",
        }
        passcode = self.headers.get("x-demo-passcode")
        if passcode:
            event_headers["x-demo-passcode"] = passcode

        try:
            lambda_response = self._invoke_lambda(_event_for(payload, event_headers))
        except urllib.error.URLError as error:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "error": "lambda_unavailable",
                    "message": f"Could not reach local Lambda emulator: {error.reason}",
                },
            )
            return

        status_code = int(lambda_response.get("statusCode", HTTPStatus.BAD_GATEWAY))
        body = lambda_response.get("body") or "{}"
        response_headers = lambda_response.get("headers") or {}
        self.send_response(status_code)
        self.send_header("content-type", response_headers.get("content-type", "application/json"))
        self.send_header("cache-control", "no-store")
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _invoke_lambda(self, event: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.lambda_url,
            data=json.dumps(event).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90.0) as response:
            return json.load(response)

    def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        encoded_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded_body)))
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(encoded_body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_LISTEN_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--lambda-url", default=DEFAULT_LAMBDA_URL)
    args = parser.parse_args()

    DemoRequestHandler.lambda_url = args.lambda_url
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"Serving demo UI at http://{args.host}:{args.port}")
    print(f"Proxying /infer to {args.lambda_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
