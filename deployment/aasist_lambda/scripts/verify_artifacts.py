#!/usr/bin/env python3
"""Verify every pinned artifact against artifact-manifest.json."""

from __future__ import annotations

import hashlib
import json
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

    checked = 0
    for section_name in ("model", "implementation"):
        for relative_path, expected in manifest[section_name]["files"].items():
            path = ROOT / relative_path
            if not path.is_file():
                raise SystemExit(f"Missing pinned artifact: {relative_path}")
            if "bytes" in expected and path.stat().st_size != expected["bytes"]:
                raise SystemExit(f"Size mismatch: {relative_path}")
            actual_hash = sha256(path)
            if actual_hash != expected["sha256"]:
                raise SystemExit(f"SHA-256 mismatch: {relative_path}")
            print(f"verified {relative_path}: {actual_hash}")
            checked += 1

    print(f"all {checked} pinned artifacts verified")


if __name__ == "__main__":
    main()

