"""Local CLI for supported audio files or predecoded NumPy waveforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inference_contract import INITIAL_DECISION_THRESHOLD, SAMPLE_RATE

from .scorer import AASISTScorer
from .service import AudioInferenceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Audio file or one-dimensional .npy waveform")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--threshold", type=float, default=INITIAL_DECISION_THRESHOLD)
    parser.add_argument("--include-windows", action="store_true")
    parser.add_argument("--ffmpeg-binary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer = AASISTScorer(threshold=args.threshold)
    if args.input.suffix.lower() == ".npy":
        waveform = np.load(args.input, allow_pickle=False)
        result = scorer.score(waveform, sample_rate=args.sample_rate)
        output = result.to_dict(include_windows=args.include_windows)
    else:
        result = AudioInferenceService(scorer).score_bytes(
            args.input.read_bytes(),
            ffmpeg_binary=args.ffmpeg_binary,
        )
        output = result.to_dict(include_windows=args.include_windows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
