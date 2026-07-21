"""Reusable waveform inference for the pinned AASIST model."""

from .errors import (
    AudioDecodeError,
    AudioDecoderUnavailableError,
    AudioTooLongError,
    InvalidThresholdError,
    InvalidWaveformError,
    UnsupportedAudioFormatError,
    UnsupportedSampleRateError,
    UploadTooLargeError,
)
from .audio import DecodedAudio, decode_audio_bytes, detect_audio_format
from .results import InferenceResult, WindowResult
from .scorer import AASISTScorer
from .service import AudioFileInferenceResult, AudioInferenceService

__all__ = [
    "AASISTScorer",
    "AudioDecodeError",
    "AudioDecoderUnavailableError",
    "AudioFileInferenceResult",
    "AudioInferenceService",
    "AudioTooLongError",
    "DecodedAudio",
    "InferenceResult",
    "InvalidThresholdError",
    "InvalidWaveformError",
    "UnsupportedAudioFormatError",
    "UnsupportedSampleRateError",
    "UploadTooLargeError",
    "WindowResult",
    "decode_audio_bytes",
    "detect_audio_format",
]
