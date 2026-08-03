"""Lightweight explanation helpers for the AASIST demo.

This intentionally ports only the small, deployable ideas from the teammate
Dash prototype: a spectrogram-like view and occlusion sensitivity. It avoids
Dash/Plotly/SHAP so the Lambda image remains close to the current inference
container.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from inference_contract import (
    BONAFIDE_CLASS_INDEX,
    SAMPLE_RATE,
    SPOOF_CLASS_INDEX,
    WINDOW_SAMPLES,
)

from .waveform import repeat_pad, validate_waveform


EXPLANATION_WINDOW_COUNT = 24
SPECTROGRAM_BANDS = 40
SPECTROGRAM_FFT_SIZE = 1024
SPECTROGRAM_HOP_LENGTH = 512
EMBEDDING_TOP_N = 20


@dataclass(frozen=True)
class OcclusionSegment:
    index: int
    start_seconds: float
    end_seconds: float
    delta_spoof_score: float


@dataclass(frozen=True)
class SpectrogramSummary:
    bands: int
    frames: int
    duration_seconds: float
    values: list[list[float]]


@dataclass(frozen=True)
class EmbeddingContribution:
    dimension: int
    contribution: float
    direction: str
    magnitude: float


@dataclass(frozen=True)
class ExplanationResult:
    method: str
    scope: str
    analyzed_duration_seconds: float
    baseline_spoof_score: float
    inference_ms: float
    embedding_contributions: tuple[EmbeddingContribution, ...]
    occlusion: tuple[OcclusionSegment, ...]
    spectrogram: SpectrogramSummary
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scope": self.scope,
            "analyzed_duration_seconds": self.analyzed_duration_seconds,
            "baseline_spoof_score": self.baseline_spoof_score,
            "inference_ms": self.inference_ms,
            "embedding_contributions": [
                asdict(contribution) for contribution in self.embedding_contributions
            ],
            "occlusion": [asdict(segment) for segment in self.occlusion],
            "spectrogram": asdict(self.spectrogram),
            "notes": list(self.notes),
        }


def explain_waveform(
    model: torch.nn.Module,
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    occlusion_segments: int = EXPLANATION_WINDOW_COUNT,
) -> ExplanationResult:
    """Return a bounded, JSON-ready explanation for one representative window."""

    started = time.perf_counter()
    validated = validate_waveform(waveform, sample_rate)
    representative = repeat_pad(validated, WINDOW_SAMPLES)
    analyzed_duration_seconds = min(validated.size, WINDOW_SAMPLES) / SAMPLE_RATE

    baseline_score, embedding = _score_and_embedding(model, representative)
    embedding_contributions = _embedding_contributions(model, embedding)
    occlusion = _occlusion_sensitivity(
        model,
        representative,
        baseline_score=baseline_score,
        n_segments=occlusion_segments,
    )
    spectrogram = _spectrogram_summary(representative)

    return ExplanationResult(
        method="spectrogram-plus-occlusion-lite",
        scope="first AASIST analysis window",
        analyzed_duration_seconds=analyzed_duration_seconds,
        baseline_spoof_score=baseline_score,
        inference_ms=(time.perf_counter() - started) * 1000.0,
        embedding_contributions=tuple(embedding_contributions),
        occlusion=tuple(occlusion),
        spectrogram=spectrogram,
        notes=(
            "Occlusion values show how the spoof score changes when each time segment is muted.",
            "Positive delta means the segment supported the spoof score; negative delta means it pushed away from spoof.",
            "Embedding contributions use the final classifier weights over AASIST latent dimensions.",
            "The spectrogram is a lightweight log-energy summary for visualization.",
        ),
    )


def _score_and_embedding(model: torch.nn.Module, waveform: np.ndarray) -> tuple[float, torch.Tensor]:
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)
        embedding, logits = model(tensor)
        score = float(torch.softmax(logits, dim=-1)[0, SPOOF_CLASS_INDEX])
        return score, embedding.squeeze(0).detach().cpu()


def _spoof_score(model: torch.nn.Module, waveform: np.ndarray) -> float:
    return _score_and_embedding(model, waveform)[0]


def _embedding_contributions(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    *,
    top_n: int = EMBEDDING_TOP_N,
) -> list[EmbeddingContribution]:
    out_layer = getattr(model, "out_layer", None)
    if out_layer is None or not hasattr(out_layer, "weight"):
        return []

    weights = out_layer.weight.detach().cpu()
    if weights.ndim != 2 or weights.shape[0] <= SPOOF_CLASS_INDEX:
        return []
    if embedding.ndim != 1 or embedding.numel() != weights.shape[1]:
        return []

    margin_weights = weights[SPOOF_CLASS_INDEX] - weights[BONAFIDE_CLASS_INDEX]
    contributions = embedding * margin_weights
    top_indices = torch.argsort(torch.abs(contributions), descending=True)[:top_n]

    results = []
    for index_tensor in top_indices:
        index = int(index_tensor.item())
        value = float(contributions[index])
        results.append(
            EmbeddingContribution(
                dimension=index,
                contribution=round(value, 6),
                direction="spoof" if value >= 0 else "bonafide",
                magnitude=round(abs(value), 6),
            )
        )
    return results



def _occlusion_sensitivity(
    model: torch.nn.Module,
    waveform: np.ndarray,
    *,
    baseline_score: float,
    n_segments: int,
) -> list[OcclusionSegment]:
    segment_len = waveform.size // n_segments
    segments: list[OcclusionSegment] = []

    for index in range(n_segments):
        start = index * segment_len
        end = waveform.size if index == n_segments - 1 else start + segment_len
        occluded = waveform.copy()
        occluded[start:end] = 0.0
        occluded_score = _spoof_score(model, occluded)
        segments.append(
            OcclusionSegment(
                index=index,
                start_seconds=round(start / SAMPLE_RATE, 3),
                end_seconds=round(end / SAMPLE_RATE, 3),
                delta_spoof_score=round(baseline_score - occluded_score, 6),
            )
        )

    return segments


def _spectrogram_summary(waveform: np.ndarray) -> SpectrogramSummary:
    if waveform.size < SPECTROGRAM_FFT_SIZE:
        waveform = repeat_pad(waveform, SPECTROGRAM_FFT_SIZE)

    starts = range(0, waveform.size - SPECTROGRAM_FFT_SIZE + 1, SPECTROGRAM_HOP_LENGTH)
    window = np.hanning(SPECTROGRAM_FFT_SIZE).astype(np.float32)
    frames = []
    for start in starts:
        chunk = waveform[start : start + SPECTROGRAM_FFT_SIZE] * window
        power = np.abs(np.fft.rfft(chunk)) ** 2
        frames.append(power)

    if not frames:
        frames.append(np.zeros(SPECTROGRAM_FFT_SIZE // 2 + 1, dtype=np.float32))

    spectrum = np.stack(frames, axis=1)
    band_edges = np.linspace(0, spectrum.shape[0], SPECTROGRAM_BANDS + 1, dtype=int)
    bands = []
    for index in range(SPECTROGRAM_BANDS):
        start, end = band_edges[index], band_edges[index + 1]
        if end <= start:
            end = start + 1
        bands.append(spectrum[start:end].mean(axis=0))

    band_matrix = np.log1p(np.stack(bands, axis=0))
    min_value = float(band_matrix.min())
    max_value = float(band_matrix.max())
    if max_value > min_value:
        band_matrix = (band_matrix - min_value) / (max_value - min_value)
    else:
        band_matrix = np.zeros_like(band_matrix)

    return SpectrogramSummary(
        bands=SPECTROGRAM_BANDS,
        frames=int(band_matrix.shape[1]),
        duration_seconds=round(waveform.size / SAMPLE_RATE, 3),
        values=np.round(band_matrix, 4).tolist(),
    )
