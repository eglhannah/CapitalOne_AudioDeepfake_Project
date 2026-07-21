"""Validation and deterministic window construction for decoded waveforms."""

from __future__ import annotations

import numpy as np

from inference_contract import (
    MAX_AUDIO_SAMPLES,
    SAMPLE_RATE,
    STRIDE_SAMPLES,
    WINDOW_SAMPLES,
)

from .errors import AudioTooLongError, InvalidWaveformError, UnsupportedSampleRateError


def validate_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != SAMPLE_RATE:
        raise UnsupportedSampleRateError(
            f"Expected {SAMPLE_RATE} Hz audio, received {sample_rate} Hz; "
            "resample before inference"
        )
    if not isinstance(waveform, np.ndarray):
        raise InvalidWaveformError("Waveform must be a NumPy array")
    if waveform.ndim != 1:
        raise InvalidWaveformError(
            f"Waveform must be mono with shape (samples,), received {waveform.shape}"
        )
    if waveform.size == 0:
        raise InvalidWaveformError("Waveform must contain at least one sample")
    if not np.issubdtype(waveform.dtype, np.floating):
        raise InvalidWaveformError("Waveform must contain normalized floating-point PCM")
    if waveform.size > MAX_AUDIO_SAMPLES:
        raise AudioTooLongError(
            f"Audio exceeds the {MAX_AUDIO_SAMPLES / SAMPLE_RATE:.0f}-second limit"
        )
    if not np.isfinite(waveform).all():
        raise InvalidWaveformError("Waveform contains NaN or infinite values")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def repeat_pad(waveform: np.ndarray, target_samples: int = WINDOW_SAMPLES) -> np.ndarray:
    """Repeat-pad a non-empty waveform to exactly ``target_samples``."""

    if waveform.size == 0:
        raise InvalidWaveformError("Cannot pad an empty waveform")
    if waveform.size >= target_samples:
        return waveform[:target_samples]
    repeats = (target_samples + waveform.size - 1) // waveform.size
    return np.tile(waveform, repeats)[:target_samples]


def window_start_samples(
    sample_count: int,
    *,
    window_samples: int = WINDOW_SAMPLES,
    stride_samples: int = STRIDE_SAMPLES,
) -> tuple[int, ...]:
    """Return starts that cover the entire waveform, including its tail."""

    if sample_count <= 0:
        raise InvalidWaveformError("Cannot window an empty waveform")
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must be positive")
    if sample_count <= window_samples:
        return (0,)

    starts = list(range(0, sample_count - window_samples + 1, stride_samples))
    tail_start = sample_count - window_samples
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return tuple(starts)


def iter_windows(waveform: np.ndarray):
    """Yield ``(start_sample, fixed_length_window)`` pairs."""

    if waveform.size < WINDOW_SAMPLES:
        yield 0, repeat_pad(waveform)
        return
    for start in window_start_samples(waveform.size):
        yield start, waveform[start : start + WINDOW_SAMPLES]

