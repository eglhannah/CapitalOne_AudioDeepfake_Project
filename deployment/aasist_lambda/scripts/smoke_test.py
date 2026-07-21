#!/usr/bin/env python3
"""Load the checkpoint and score one deterministic synthetic waveform."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_contract import (  # noqa: E402
    INITIAL_DECISION_THRESHOLD,
    MODEL_NAME,
    MODEL_REVISION,
    SPOOF_CLASS_INDEX,
    WINDOW_SAMPLES,
)
from model_loader import load_model  # noqa: E402


def main() -> None:
    model, checkpoint_metadata = load_model()
    waveform = np.random.default_rng(1234).normal(
        loc=0.0,
        scale=0.01,
        size=WINDOW_SAMPLES,
    ).astype(np.float32)

    with torch.inference_mode():
        _, logits = model(torch.from_numpy(waveform).unsqueeze(0))
        probabilities = torch.softmax(logits, dim=-1)

    spoof_score = float(probabilities[0, SPOOF_CLASS_INDEX])
    result = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "checkpoint": checkpoint_metadata,
        "input": {
            "kind": "deterministic_gaussian_noise",
            "numpy_seed": 1234,
            "samples": WINDOW_SAMPLES,
        },
        "logits": [float(value) for value in logits[0]],
        "spoof_score": spoof_score,
        "initial_threshold": INITIAL_DECISION_THRESHOLD,
        "initial_classification": (
            "spoof" if spoof_score >= INITIAL_DECISION_THRESHOLD else "bonafide"
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

