"""AWS Lambda Function URL adapter for in-memory AASIST inference."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import os
import time
from typing import Any

from aasist_inference import AudioInferenceService
from aasist_inference.errors import (
    AudioDecodeError,
    AudioTooLongError,
    InferenceInputError,
    UnsupportedAudioFormatError,
    UploadTooLargeError,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Lambda imports this module during environment initialization. Keeping the
# service global means warm invocations reuse the already-loaded model.
_INITIALIZATION_STARTED = time.perf_counter()
_SERVICE = AudioInferenceService()
_INITIALIZATION_MS = (time.perf_counter() - _INITIALIZATION_STARTED) * 1000.0
_COLD_START = True


def _headers(event: dict[str, Any]) -> dict[str, str]:
    headers = event.get("headers") or {}
    if not isinstance(headers, dict):
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}


# def _cors_headers(event: dict[str, Any]) -> dict[str, str]:
#     configured_origin = os.environ.get("DEMO_ALLOWED_ORIGIN", "*")
#     request_origin = _headers(event).get("origin")
#     allowed_origin = request_origin if configured_origin == "request" and request_origin else configured_origin
#     return {
#         "access-control-allow-origin": allowed_origin,
#         "access-control-allow-methods": "POST,OPTIONS",
#         "access-control-allow-headers": "content-type,x-demo-passcode",
#         "access-control-max-age": "3600",
#         "vary": "origin",
#     }


def _response(status_code: int, body: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "content-type": "application/json",
            "cache-control": "no-store",
            # **_cors_headers(event),
        },
        "body": json.dumps(body, separators=(",", ":")),
        "isBase64Encoded": False,
    }


def _request_id(event: dict[str, Any], context: Any) -> str:
    context_id = getattr(context, "aws_request_id", None)
    return str(context_id or event.get("requestContext", {}).get("requestId") or "unknown")


def _method(event: dict[str, Any]) -> str:
    request_context = event.get("requestContext") or {}
    http_context = request_context.get("http") or {}
    return str(http_context.get("method") or event.get("httpMethod") or "").upper()


def _is_authorized(event: dict[str, Any]) -> bool:
    expected_passphrase = os.environ.get("DEMO_PASSPHRASE")
    if not expected_passphrase:
        return True
    supplied_passphrase = _headers(event).get("x-demo-passcode", "")
    return hmac.compare_digest(supplied_passphrase, expected_passphrase)


def _decode_event_body(event: dict[str, Any]) -> bytes:
    body = event.get("body")
    if not isinstance(body, str) or not body:
        raise ValueError("Request body must contain an audio file")
    if event.get("isBase64Encoded") is True:
        try:
            return base64.b64decode(body, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Request body is not valid base64") from error
    return body.encode("utf-8")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    global _COLD_START

    request_id = _request_id(event, context)
    cold_start = _COLD_START
    _COLD_START = False
    started = time.perf_counter()
    method = _method(event)

    if method != "POST":
        if method == "OPTIONS":
            return _response(
                200,
                {
                    "ok": True,
                    "request_id": request_id,
                },
                event,
            )
        return _response(
            405,
            {
                "error": "method_not_allowed",
                "message": "Use POST with a binary audio request body",
                "request_id": request_id,
            },
            event,
        )

    if not _is_authorized(event):
        return _response(
            401,
            {
                "error": "unauthorized",
                "message": "Missing or incorrect demo passcode",
                "request_id": request_id,
            },
            event,
        )

    try:
        payload = _decode_event_body(event)
        result = _SERVICE.score_bytes(payload)
        response_body = result.to_dict()
        response_body.update(
            {
                "request_id": request_id,
                "cold_start": cold_start,
                "initialization_ms": _INITIALIZATION_MS if cold_start else 0.0,
            }
        )
        LOGGER.info(
            "inference_complete request_id=%s format=%s bytes=%d windows=%d duration_ms=%.2f",
            request_id,
            result.detected_format,
            result.encoded_size_bytes,
            result.inference.window_count,
            (time.perf_counter() - started) * 1000.0,
        )
        return _response(200, response_body, event)
    except UploadTooLargeError as error:
        return _response(
            413,
            {"error": "upload_too_large", "message": str(error), "request_id": request_id},
            event,
        )
    except UnsupportedAudioFormatError as error:
        return _response(
            415,
            {"error": "unsupported_audio_format", "message": str(error), "request_id": request_id},
            event,
        )
    except (AudioDecodeError, AudioTooLongError) as error:
        return _response(
            422,
            {"error": "unprocessable_audio", "message": str(error), "request_id": request_id},
            event,
        )
    except InferenceInputError as error:
        return _response(
            400,
            {"error": "invalid_audio", "message": str(error), "request_id": request_id},
            event,
        )
    except ValueError as error:
        return _response(
            400,
            {"error": "invalid_request", "message": str(error), "request_id": request_id},
            event,
        )
    except Exception:
        LOGGER.exception("unexpected_inference_error request_id=%s", request_id)
        return _response(
            500,
            {
                "error": "internal_error",
                "message": "Inference failed unexpectedly",
                "request_id": request_id,
            },
            event,
        )
