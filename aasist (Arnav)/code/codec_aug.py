"""Codec augmentation by random substitution from a pre-computed cache.

Reads codec-round-tripped copies of train files produced by
precompute_codec_train.py and randomly substitutes one for the clean
waveform during training.

Apply only during TRAINING. Do not apply at evaluation time.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

DEFAULT_BASE = Path(f"/scratch/{os.environ.get('USER', 'unknown')}/aasist/data/LA_codec_train")


class CodecAugment:
    """Callable that randomly substitutes a codec-augmented waveform.

    Args:
        codec_base: Directory containing <codec>/<filename>.wav files.
        codecs: Codec names to choose from. Each must be a subdirectory of codec_base.
        p_codec: Probability of substituting a codec version on a given call.
        seed: Optional seed for the internal RNG (omit for fresh per-worker state).
    """

    def __init__(
        self,
        codec_base: Path | str | None = None,
        codecs: Sequence[str] = ("alaw", "ulaw", "g722", "opus"),
        p_codec: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.codec_base = Path(codec_base) if codec_base else DEFAULT_BASE
        self.codecs = tuple(codecs)
        self.p_codec = p_codec
        self.rng = np.random.default_rng(seed)
        self.missing_logged: set[str] = set()

    def maybe_substitute(
        self,
        clean_path: str | Path,
        clean_waveform: np.ndarray,
        target_len: int | None = None,
    ) -> tuple[np.ndarray, str]:
        """Maybe replace clean_waveform with a codec-round-tripped version.

        Args:
            clean_path: Path to the original clean .flac (only the stem is used).
            clean_waveform: The clean waveform already loaded by the dataset.
            target_len: If given, pad/truncate output to this length. Otherwise
                match len(clean_waveform).

        Returns:
            (waveform, codec_used). codec_used is "clean" if no substitution,
            "clean_fallback" if the codec file was missing, or the codec name.
        """
        L = target_len if target_len is not None else len(clean_waveform)

        if self.rng.random() > self.p_codec:
            return self._fit_length(clean_waveform, L), "clean"

        codec = str(self.rng.choice(self.codecs))
        stem = Path(clean_path).stem
        codec_path = self.codec_base / codec / f"{stem}.wav"

        if not codec_path.exists():
            key = f"{codec}/{stem}"
            if key not in self.missing_logged and len(self.missing_logged) < 20:
                self.missing_logged.add(key)
            return self._fit_length(clean_waveform, L), "clean_fallback"

        try:
            wav, _ = sf.read(str(codec_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav[:, 0]
            return self._fit_length(wav.astype(np.float32), L), codec
        except Exception:
            return self._fit_length(clean_waveform, L), "clean_fallback"

    @staticmethod
    def _fit_length(x: np.ndarray, L: int) -> np.ndarray:
        if len(x) == L:
            return x.astype(np.float32, copy=False)
        if len(x) > L:
            return x[:L].astype(np.float32, copy=False)
        return np.pad(x, (0, L - len(x))).astype(np.float32, copy=False)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake = rng.standard_normal(64600).astype(np.float32) * 0.1
    aug = CodecAugment(codec_base="/nonexistent", p_codec=1.0)
    out, used = aug.maybe_substitute("LA_T_1234567.flac", fake)
    print(f"Fallback path test: used={used} shape={out.shape} dtype={out.dtype}")
    assert out.shape == fake.shape and out.dtype == np.float32
    print("OK")
