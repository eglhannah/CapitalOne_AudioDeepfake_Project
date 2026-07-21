from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_contract import (  # noqa: E402
    BONAFIDE_CLASS_INDEX,
    SAMPLE_RATE,
    SPOOF_CLASS_INDEX,
    WINDOW_SAMPLES,
)
from model_loader import load_model, read_training_config  # noqa: E402


class ModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "tests" / "golden-synthetic.json").open(encoding="utf-8") as file:
            cls.golden = json.load(file)
        cls.model, cls.metadata = load_model()

    def test_training_config_matches_inference_contract(self) -> None:
        config = read_training_config()
        self.assertEqual(config["model"]["nb_samp"], WINDOW_SAMPLES)
        self.assertEqual(SAMPLE_RATE, 16_000)
        self.assertEqual(BONAFIDE_CLASS_INDEX, 0)
        self.assertEqual(SPOOF_CLASS_INDEX, 1)

    def test_checkpoint_metadata(self) -> None:
        self.assertEqual(self.metadata["epoch"], self.golden["checkpoint_epoch"])
        self.assertEqual(self.metadata["parameter_count"], self.golden["parameter_count"])
        self.assertEqual(self.metadata["metrics"]["threshold"], 0.5)

    def test_deterministic_forward_pass(self) -> None:
        waveform = np.random.default_rng(self.golden["numpy_seed"]).normal(
            loc=0.0,
            scale=0.01,
            size=self.golden["samples"],
        ).astype(np.float32)
        with torch.inference_mode():
            _, logits = self.model(torch.from_numpy(waveform).unsqueeze(0))
            score = torch.softmax(logits, dim=-1)[0, SPOOF_CLASS_INDEX].item()

        tolerance = self.golden["absolute_tolerance"]
        for actual, expected in zip(logits[0].tolist(), self.golden["logits"]):
            self.assertAlmostEqual(actual, expected, delta=tolerance)
        self.assertAlmostEqual(score, self.golden["spoof_score"], delta=tolerance)


if __name__ == "__main__":
    unittest.main()

