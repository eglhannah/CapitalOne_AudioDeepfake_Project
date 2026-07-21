#!/usr/bin/env python3
"""Fetch pinned model artifacts and verify them before installation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with (ROOT / "artifact-manifest.json").open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    model = manifest["model"]
    base_url = (
        f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}"
    )
    for relative_path, expected in model["files"].items():
        destination = ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{expected['source_file']}"

        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp_file:
            temporary_path = Path(temp_file.name)
        try:
            print(f"fetching {url}")
            urllib.request.urlretrieve(url, temporary_path)
            if temporary_path.stat().st_size != expected["bytes"]:
                raise SystemExit(f"Size mismatch while fetching {relative_path}")
            if sha256(temporary_path) != expected["sha256"]:
                raise SystemExit(f"SHA-256 mismatch while fetching {relative_path}")
            temporary_path.replace(destination)
            print(f"installed {relative_path}")
        finally:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

