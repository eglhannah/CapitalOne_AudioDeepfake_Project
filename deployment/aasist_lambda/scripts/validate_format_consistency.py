#!/usr/bin/env python3
"""Transcode private samples in memory and compare model scores by format."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aasist_inference import AudioInferenceService  # noqa: E402
from aasist_inference.audio import resolve_ffmpeg_binary  # noqa: E402

FORMAT_ARGUMENTS = {
    "wav": ["-c:a", "pcm_s16le", "-f", "wav"],
    "flac": ["-c:a", "flac", "-f", "flac"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
    "m4a": [
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
    ],
    "ogg": ["-c:a", "libvorbis", "-q:a", "5", "-f", "ogg"],
    "webm": ["-c:a", "libopus", "-b:a", "96k", "-f", "webm"],
}


def transcode(ffmpeg: str, source: bytes, arguments: list[str]) -> bytes:
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            *arguments,
            "pipe:1",
        ],
        input=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("Private sample transcoding failed")
    return completed.stdout


def main() -> None:
    ffmpeg = resolve_ffmpeg_binary()
    service = AudioInferenceService()
    results = []
    for path in sorted((ROOT / "local_samples").glob("*.flac")):
        source = path.read_bytes()
        reference_score = service.score_bytes(source, ffmpeg_binary=ffmpeg).inference.spoof_score
        format_results = []
        for format_name, arguments in FORMAT_ARGUMENTS.items():
            encoded = transcode(ffmpeg, source, arguments)
            scored = service.score_bytes(encoded, ffmpeg_binary=ffmpeg)
            format_results.append(
                {
                    "format": format_name,
                    "score": scored.inference.spoof_score,
                    "absolute_delta_from_source": abs(
                        scored.inference.spoof_score - reference_score
                    ),
                    "encoded_size_bytes": len(encoded),
                }
            )
        results.append(
            {
                "filename": path.name,
                "source_flac_score": reference_score,
                "formats": format_results,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
