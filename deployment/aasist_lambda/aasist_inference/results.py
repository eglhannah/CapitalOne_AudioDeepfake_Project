"""Structured, JSON-ready inference results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WindowResult:
    index: int
    start_sample: int
    end_sample: int
    spoof_score: float
    inference_ms: float


@dataclass(frozen=True)
class InferenceResult:
    spoof_score: float
    classification: str
    threshold: float
    window_count: int
    audio_duration_seconds: float
    inference_ms: float
    model_name: str
    model_revision: str
    checkpoint_epoch: int | None
    windows: tuple[WindowResult, ...]

    def to_dict(self, *, include_windows: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "spoof_score": self.spoof_score,
            "classification": self.classification,
            "threshold": self.threshold,
            "window_count": self.window_count,
            "audio_duration_seconds": self.audio_duration_seconds,
            "inference_ms": self.inference_ms,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "checkpoint_epoch": self.checkpoint_epoch,
        }
        if include_windows:
            result["windows"] = [asdict(window) for window in self.windows]
        return result

