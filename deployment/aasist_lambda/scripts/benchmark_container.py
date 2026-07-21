#!/usr/bin/env python3
"""Measure local RIE round-trip and reported model latency by clip length."""

from __future__ import annotations

import io
import json
import struct
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from container_client import function_url_event, invoke, wait_until_ready  # noqa: E402


def silence_wav(duration_seconds: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(struct.pack("<h", 0) * (duration_seconds * 16_000))
    return output.getvalue()


def main() -> None:
    wait_until_ready()
    rows = []
    for duration_seconds in (1, 10, 30):
        event = function_url_event(silence_wav(duration_seconds))
        started = time.perf_counter()
        response = invoke(event)
        round_trip_ms = (time.perf_counter() - started) * 1000.0
        if response["statusCode"] != 200:
            raise AssertionError(response)
        body = json.loads(response["body"])
        rows.append(
            {
                "duration_seconds": duration_seconds,
                "window_count": body["window_count"],
                "round_trip_ms": round_trip_ms,
                "model_inference_ms": body["inference_ms"],
                "cold_start": body["cold_start"],
                "initialization_ms": body["initialization_ms"],
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

