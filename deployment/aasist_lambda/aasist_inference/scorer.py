"""Core CPU inference service for decoded 16 kHz mono waveforms."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from inference_contract import (
    INITIAL_DECISION_THRESHOLD,
    MODEL_NAME,
    MODEL_REVISION,
    SAMPLE_RATE,
    SPOOF_CLASS_INDEX,
)
from model_loader import load_model

from .errors import InvalidThresholdError
from .results import InferenceResult, WindowResult
from .waveform import iter_windows, validate_waveform


class AASISTScorer:
    """Own one evaluated CPU model and reuse it across inference calls."""

    def __init__(
        self,
        *,
        model: torch.nn.Module | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        threshold: float = INITIAL_DECISION_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise InvalidThresholdError("Threshold must be between 0 and 1")
        if model is None:
            model, loaded_metadata = load_model()
            checkpoint_metadata = loaded_metadata

        self._model = model.to("cpu")
        self._model.eval()
        self._metadata = checkpoint_metadata or {}
        self._threshold = float(threshold)

    @property
    def threshold(self) -> float:
        return self._threshold

    def score(self, waveform: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> InferenceResult:
        validated = validate_waveform(waveform, sample_rate)
        window_results: list[WindowResult] = []

        with torch.inference_mode():
            for index, (start, window) in enumerate(iter_windows(validated)):
                tensor = torch.from_numpy(window).unsqueeze(0)
                started = time.perf_counter()
                _, logits = self._model(tensor)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                if logits.shape != (1, 2):
                    raise RuntimeError(
                        f"Expected model logits with shape (1, 2), received {tuple(logits.shape)}"
                    )
                spoof_score = float(torch.softmax(logits, dim=-1)[0, SPOOF_CLASS_INDEX])
                window_results.append(
                    WindowResult(
                        index=index,
                        start_sample=start,
                        # Report coverage in the original waveform, not the
                        # synthetic repeat-padding added for short clips.
                        end_sample=min(start + window.size, validated.size),
                        spoof_score=spoof_score,
                        inference_ms=elapsed_ms,
                    )
                )

        aggregate_score = sum(window.spoof_score for window in window_results) / len(
            window_results
        )
        return InferenceResult(
            spoof_score=aggregate_score,
            classification="spoof" if aggregate_score >= self._threshold else "bonafide",
            threshold=self._threshold,
            window_count=len(window_results),
            audio_duration_seconds=validated.size / SAMPLE_RATE,
            inference_ms=sum(window.inference_ms for window in window_results),
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            checkpoint_epoch=self._metadata.get("epoch"),
            windows=tuple(window_results),
        )
