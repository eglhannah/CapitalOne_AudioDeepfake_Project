"""Compose bounded file decoding with the model-level waveform scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio import decode_audio_bytes
from .results import InferenceResult
from .scorer import AASISTScorer


@dataclass(frozen=True)
class AudioFileInferenceResult:
    detected_format: str
    encoded_size_bytes: int
    inference: InferenceResult

    def to_dict(self, *, include_windows: bool = False) -> dict[str, Any]:
        result = self.inference.to_dict(include_windows=include_windows)
        result["input"] = {
            "detected_format": self.detected_format,
            "encoded_size_bytes": self.encoded_size_bytes,
        }
        return result


class AudioInferenceService:
    def __init__(self, scorer: AASISTScorer | None = None) -> None:
        self._scorer = scorer or AASISTScorer()

    def score_bytes(
        self,
        payload: bytes,
        *,
        ffmpeg_binary: str | None = None,
    ) -> AudioFileInferenceResult:
        decoded = decode_audio_bytes(payload, ffmpeg_binary=ffmpeg_binary)
        inference = self._scorer.score(decoded.waveform, sample_rate=decoded.sample_rate)
        return AudioFileInferenceResult(
            detected_format=decoded.detected_format,
            encoded_size_bytes=decoded.encoded_size_bytes,
            inference=inference,
        )

