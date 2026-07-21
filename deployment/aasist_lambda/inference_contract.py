"""Stable model-level contract for AASIST v3 deployment.

Audio decoding, resampling, and request handling belong to later phases. This
module deliberately captures only the assumptions required by the checkpoint.
"""

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 64_600
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
STRIDE_SAMPLES = 32_000
STRIDE_SECONDS = STRIDE_SAMPLES / SAMPLE_RATE
MAX_AUDIO_SECONDS = 30.0
MAX_AUDIO_SAMPLES = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
AUDIO_DECODE_TIMEOUT_SECONDS = 15.0

BONAFIDE_CLASS_INDEX = 0
SPOOF_CLASS_INDEX = 1
INITIAL_DECISION_THRESHOLD = 0.5

MODEL_NAME = "aasist-v3-codecaugment"
MODEL_REVISION = "sha256:36e27b4b2032c0a7448f7c9dab2db89efd607013b8596da2b7be8419814b83d0"
