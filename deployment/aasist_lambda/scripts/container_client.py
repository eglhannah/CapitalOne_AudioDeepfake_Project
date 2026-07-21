"""Small standard-library client for the local Lambda invocation endpoint."""

from __future__ import annotations

import base64
import json
import socket
import time
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:9000/2015-03-31/functions/function/invocations"


def function_url_event(payload: bytes, *, method: str = "POST") -> dict[str, Any]:
    return {
        "version": "2.0",
        "requestContext": {
            "requestId": "local-container-request",
            "http": {"method": method},
        },
        "body": base64.b64encode(payload).decode("ascii"),
        "isBase64Encoded": True,
    }


def invoke(event: dict[str, Any], *, url: str = DEFAULT_URL, timeout: float = 90.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(event).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_until_ready(*, url: str = DEFAULT_URL, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 9000), timeout=1.0):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("Local Lambda emulator did not become ready")
