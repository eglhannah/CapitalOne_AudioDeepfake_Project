from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aasist_inference.errors import InvalidThresholdError  # noqa: E402
from aasist_inference.scorer import AASISTScorer  # noqa: E402
from inference_contract import WINDOW_SAMPLES  # noqa: E402


class SequenceProbabilityModel(torch.nn.Module):
    """Return a configured spoof probability on each successive invocation."""

    def __init__(self, probabilities: list[float]) -> None:
        super().__init__()
        self.probabilities = probabilities
        self.call_count = 0

    def forward(self, waveform):
        probability = self.probabilities[self.call_count]
        self.call_count += 1
        logits = torch.tensor(
            [[math.log(1.0 - probability), math.log(probability)]],
            dtype=torch.float32,
        )
        return torch.empty((1, 0)), logits


class ScorerTest(unittest.TestCase):
    def test_rejects_invalid_threshold(self) -> None:
        for threshold in (-0.01, 1.01):
            with self.subTest(threshold=threshold):
                with self.assertRaises(InvalidThresholdError):
                    AASISTScorer(model=SequenceProbabilityModel([0.5]), threshold=threshold)

    def test_averages_windows_and_classifies_score_at_threshold_as_spoof(self) -> None:
        model = SequenceProbabilityModel([0.2, 0.8])
        result = AASISTScorer(model=model).score(
            np.zeros(5 * 16_000, dtype=np.float32)
        )
        self.assertAlmostEqual(result.spoof_score, 0.5, places=6)
        self.assertEqual(result.classification, "spoof")
        self.assertEqual(result.window_count, 2)
        self.assertEqual(result.windows[-1].end_sample, 5 * 16_000)

    def test_result_serialization_hides_window_details_by_default(self) -> None:
        result = AASISTScorer(model=SequenceProbabilityModel([0.25])).score(
            np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        )
        self.assertNotIn("windows", result.to_dict())
        self.assertEqual(len(result.to_dict(include_windows=True)["windows"]), 1)

    def test_short_clip_reports_original_coverage_not_padding(self) -> None:
        result = AASISTScorer(model=SequenceProbabilityModel([0.25])).score(
            np.zeros(16_000, dtype=np.float32)
        )
        self.assertEqual(result.windows[0].start_sample, 0)
        self.assertEqual(result.windows[0].end_sample, 16_000)


class RealModelScorerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "tests" / "golden-synthetic.json").open(encoding="utf-8") as file:
            cls.golden = json.load(file)
        cls.scorer = AASISTScorer()

    def _golden_waveform(self) -> np.ndarray:
        return np.random.default_rng(self.golden["numpy_seed"]).normal(
            loc=0.0,
            scale=0.01,
            size=self.golden["samples"],
        ).astype(np.float32)

    def test_scorer_matches_phase_one_golden_output(self) -> None:
        result = self.scorer.score(self._golden_waveform())
        self.assertAlmostEqual(
            result.spoof_score,
            self.golden["spoof_score"],
            delta=self.golden["absolute_tolerance"],
        )
        self.assertEqual(result.window_count, 1)
        self.assertEqual(result.checkpoint_epoch, self.golden["checkpoint_epoch"])

    def test_repeated_inference_is_stable(self) -> None:
        waveform = self._golden_waveform()
        first = self.scorer.score(waveform).spoof_score
        second = self.scorer.score(waveform).spoof_score
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
