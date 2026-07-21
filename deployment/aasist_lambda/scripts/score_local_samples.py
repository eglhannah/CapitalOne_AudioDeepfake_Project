#!/usr/bin/env python3
"""Score private local samples without copying audio or results into Git."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aasist_inference import AudioInferenceService  # noqa: E402


def main() -> None:
    sample_directory = ROOT / "local_samples"
    files = sorted(path for path in sample_directory.iterdir() if path.is_file())
    if not files:
        raise SystemExit(f"No private samples found in {sample_directory}")

    service = AudioInferenceService()
    results = []
    for path in files:
        result = service.score_bytes(path.read_bytes())
        output = result.to_dict()
        output["filename"] = path.name
        results.append(output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

