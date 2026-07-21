#!/usr/bin/env python3
"""Measure local scorer initialization and warm CPU inference latency."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aasist_inference import AASISTScorer  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    scorer = AASISTScorer()
    initialization_ms = (time.perf_counter() - started) * 1000.0

    rows = []
    for duration_seconds in (1, 4, 10, 30):
        waveform = np.random.default_rng(duration_seconds).normal(
            0.0, 0.01, int(duration_seconds * 16_000)
        ).astype(np.float32)
        started = time.perf_counter()
        result = scorer.score(waveform)
        total_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "duration_seconds": duration_seconds,
                "window_count": result.window_count,
                "model_inference_ms": result.inference_ms,
                "total_score_ms": total_ms,
            }
        )

    print(json.dumps({"initialization_ms": initialization_ms, "runs": rows}, indent=2))


if __name__ == "__main__":
    main()

