"""Bounded, in-memory decoding for common presentation audio formats."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_contract import (
    AUDIO_DECODE_TIMEOUT_SECONDS,
    MAX_AUDIO_SAMPLES,
    MAX_AUDIO_SECONDS,
    MAX_UPLOAD_BYTES,
    SAMPLE_RATE,
)

from .errors import (
    AudioDecodeError,
    AudioDecoderUnavailableError,
    AudioTooLongError,
    UnsupportedAudioFormatError,
    UploadTooLargeError,
)


@dataclass(frozen=True)
class DecodedAudio:
    waveform: np.ndarray
    sample_rate: int
    detected_format: str
    encoded_size_bytes: int

    @property
    def duration_seconds(self) -> float:
        return self.waveform.size / self.sample_rate


def detect_audio_format(payload: bytes) -> str:
    """Identify an allowed container from magic bytes, never its extension."""

    if payload.startswith(b"fLaC"):
        return "flac"
    if len(payload) >= 12 and payload[:4] in (b"RIFF", b"RF64") and payload[8:12] == b"WAVE":
        return "wav"
    if payload.startswith(b"OggS"):
        return "ogg"
    if payload.startswith(b"\x1aE\xdf\xa3"):
        return "webm"
    if b"ftyp" in payload[4:64]:
        return "m4a"
    if payload.startswith(b"ID3"):
        return "mp3"
    if len(payload) >= 2 and payload[0] == 0xFF and payload[1] & 0xE0 == 0xE0:
        return "mp3-or-aac"
    raise UnsupportedAudioFormatError(
        "Unsupported audio signature; allowed formats are WAV, FLAC, MP3, "
        "M4A/AAC, OGG, and WebM"
    )


def resolve_ffmpeg_binary(explicit_binary: str | os.PathLike[str] | None = None) -> str:
    candidates = [explicit_binary, os.environ.get("FFMPEG_BINARY"), shutil.which("ffmpeg")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    # Local-development fallback only. The Lambda image will install system
    # FFmpeg and will not depend on imageio-ffmpeg.
    try:
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    except (ImportError, RuntimeError):
        pass
    raise AudioDecoderUnavailableError("No executable FFmpeg binary is available")


def decode_audio_bytes(
    payload: bytes,
    *,
    ffmpeg_binary: str | os.PathLike[str] | None = None,
) -> DecodedAudio:
    """Decode an upload to mono 16 kHz float32 PCM without writing it to disk."""

    if not isinstance(payload, bytes):
        raise TypeError("Encoded audio payload must be bytes")
    if not payload:
        raise AudioDecodeError("Audio upload is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(
            f"Audio upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit"
        )
    detected_format = detect_audio_format(payload)
    executable = resolve_ffmpeg_binary(ffmpeg_binary)

    # Decode only a few samples past the accepted limit. This bounds stdout to
    # roughly 1.9 MiB even when a small compressed upload expands to hours.
    bounded_duration = MAX_AUDIO_SECONDS + 0.01
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-t",
        f"{bounded_duration:.2f}",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=AUDIO_DECODE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise AudioDecodeError("Audio decoding exceeded the time limit") from error

    if completed.returncode != 0:
        # FFmpeg stderr can contain attacker-controlled metadata. Do not expose
        # or log it in a client-facing error.
        raise AudioDecodeError(f"FFmpeg could not decode the {detected_format} upload")
    if not completed.stdout or len(completed.stdout) % np.dtype("<f4").itemsize:
        raise AudioDecodeError("FFmpeg returned invalid or empty PCM audio")

    waveform = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
    if waveform.size > MAX_AUDIO_SAMPLES:
        raise AudioTooLongError(f"Decoded audio exceeds the {MAX_AUDIO_SECONDS:.0f}-second limit")
    if not np.isfinite(waveform).all():
        raise AudioDecodeError("Decoded audio contains non-finite samples")
    return DecodedAudio(
        waveform=waveform,
        sample_rate=SAMPLE_RATE,
        detected_format=detected_format,
        encoded_size_bytes=len(payload),
    )

