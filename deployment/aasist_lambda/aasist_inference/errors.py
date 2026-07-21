"""Typed input errors that later API layers can map to HTTP responses."""


class InferenceInputError(ValueError):
    """Base class for client-correctable inference input errors."""


class InvalidWaveformError(InferenceInputError):
    """The decoded waveform does not satisfy the model input contract."""


class UnsupportedSampleRateError(InferenceInputError):
    """The caller has not resampled audio to the model's sample rate."""


class AudioTooLongError(InferenceInputError):
    """The decoded waveform exceeds the configured inference duration."""


class InvalidThresholdError(InferenceInputError):
    """The classification threshold is outside the probability range."""


class UnsupportedAudioFormatError(InferenceInputError):
    """The upload signature does not identify an allowed audio container."""


class UploadTooLargeError(InferenceInputError):
    """The encoded upload exceeds the request-size limit."""


class AudioDecodeError(InferenceInputError):
    """FFmpeg could not decode the bounded audio payload."""


class AudioDecoderUnavailableError(RuntimeError):
    """No usable FFmpeg executable is available in the runtime."""
