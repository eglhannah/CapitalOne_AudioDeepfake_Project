from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aasist_inference.errors import (  # noqa: E402
    AudioTooLongError,
    InvalidWaveformError,
    UnsupportedSampleRateError,
)
from aasist_inference.waveform import (  # noqa: E402
    iter_windows,
    repeat_pad,
    validate_waveform,
    window_start_samples,
)
from inference_contract import MAX_AUDIO_SAMPLES, WINDOW_SAMPLES  # noqa: E402


class WaveformTest(unittest.TestCase):
    def test_repeat_pad_uses_original_sequence(self) -> None:
        padded = repeat_pad(np.array([1.0, 2.0, 3.0], dtype=np.float32), target_samples=8)
        np.testing.assert_array_equal(padded, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0])

    def test_exact_window_has_one_start(self) -> None:
        self.assertEqual(window_start_samples(WINDOW_SAMPLES), (0,))

    def test_one_sample_over_window_covers_tail(self) -> None:
        self.assertEqual(window_start_samples(WINDOW_SAMPLES + 1), (0, 1))

    def test_five_ten_and_thirty_second_window_counts(self) -> None:
        self.assertEqual(len(window_start_samples(5 * 16_000)), 2)
        self.assertEqual(len(window_start_samples(10 * 16_000)), 4)
        self.assertEqual(len(window_start_samples(30 * 16_000)), 14)

    def test_last_window_always_reaches_final_sample(self) -> None:
        for sample_count in (80_000, 160_000, 480_000):
            final_start = window_start_samples(sample_count)[-1]
            self.assertEqual(final_start + WINDOW_SAMPLES, sample_count)

    def test_short_audio_produces_fixed_length_window(self) -> None:
        waveform = np.ones(16_000, dtype=np.float32)
        windows = list(iter_windows(waveform))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[0][1].size, WINDOW_SAMPLES)

    def test_validation_rejects_empty_stereo_integer_and_nonfinite_inputs(self) -> None:
        invalid_waveforms = (
            np.array([], dtype=np.float32),
            np.zeros((2, 100), dtype=np.float32),
            np.zeros(100, dtype=np.int16),
            np.array([0.0, np.nan], dtype=np.float32),
            np.array([0.0, np.inf], dtype=np.float32),
        )
        for waveform in invalid_waveforms:
            with self.subTest(shape=waveform.shape, dtype=waveform.dtype):
                with self.assertRaises(InvalidWaveformError):
                    validate_waveform(waveform, 16_000)

    def test_validation_rejects_wrong_sample_rate(self) -> None:
        with self.assertRaises(UnsupportedSampleRateError):
            validate_waveform(np.zeros(100, dtype=np.float32), 44_100)

    def test_validation_rejects_audio_over_limit(self) -> None:
        with self.assertRaises(AudioTooLongError):
            validate_waveform(np.zeros(MAX_AUDIO_SAMPLES + 1, dtype=np.float32), 16_000)


if __name__ == "__main__":
    unittest.main()

