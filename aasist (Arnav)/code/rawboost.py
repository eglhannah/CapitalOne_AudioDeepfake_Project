"""RawBoost data augmentation for raw-waveform anti-spoofing training.

Implements the four augmentation families from:
    Tak, H., Kamble, M., Patino, J., Todisco, M., & Evans, N. (2022).
    "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic
    Speaker Verification Anti-Spoofing." ICASSP 2022. arXiv:2111.04433

Reference implementation: https://github.com/TakHemlata/RawBoost-antispoofing

Each augmentation is applied independently with its own probability:
    1. Linear convolutive noise (LnL) — FIR filter convolution
    2. Impulsive signal-dependent noise (ISD) — sparse impulse spikes
    3. Stationary signal-independent noise (SSI) — broadband Gaussian noise
    4. Codec simulation — μ-law / A-law companding (8-bit quantization)

Apply only during TRAINING. Do not apply at evaluation time.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy import signal


class RawBoostAugment:
    """Callable that applies RawBoost augmentation to a 1D waveform.

    Args:
        sample_rate: Audio sample rate in Hz (default 16000).

        # Linear convolutive noise (LnL)
        p_lnl: Probability of applying LnL augmentation each call.
        lnl_n_taps_range: Range for FIR filter length (number of taps).
        lnl_gain_db_range: Range for gain (dB) of the filtered signal added back
            to the original (negative = quieter than original).

        # Impulsive signal-dependent noise (ISD)
        p_isd: Probability of applying ISD augmentation.
        isd_n_impulses_range: Range for number of impulse spikes added.
        isd_amplitude_range: Range for amplitude of each impulse.

        # Stationary signal-independent noise (SSI)
        p_ssi: Probability of applying SSI augmentation.
        ssi_snr_db_range: Range for SNR (dB) of the added Gaussian noise.

        # Codec simulation
        p_codec: Probability of applying codec augmentation.
        codec_choices: Tuple of codec names to randomly select from.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        # LnL
        p_lnl: float = 0.5,
        lnl_n_taps_range: Tuple[int, int] = (5, 20),
        lnl_gain_db_range: Tuple[float, float] = (-15.0, -5.0),
        # ISD
        p_isd: float = 0.5,
        isd_n_impulses_range: Tuple[int, int] = (1, 5),
        isd_amplitude_range: Tuple[float, float] = (0.01, 0.1),
        # SSI
        p_ssi: float = 0.5,
        ssi_snr_db_range: Tuple[float, float] = (10.0, 30.0),
        # Codec
        p_codec: float = 0.5,
        codec_choices: Tuple[str, ...] = ("mulaw", "alaw"),
        # RNG
        seed: int | None = None,
    ):
        self.sample_rate = sample_rate

        self.p_lnl = p_lnl
        self.lnl_n_taps_range = lnl_n_taps_range
        self.lnl_gain_db_range = lnl_gain_db_range

        self.p_isd = p_isd
        self.isd_n_impulses_range = isd_n_impulses_range
        self.isd_amplitude_range = isd_amplitude_range

        self.p_ssi = p_ssi
        self.ssi_snr_db_range = ssi_snr_db_range

        self.p_codec = p_codec
        self.codec_choices = codec_choices

        # Use a private RNG for reproducibility under torch DataLoader workers.
        # If seed is None, draw from global numpy.random.
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply RawBoost to a 1D float32 waveform.

        Args:
            x: 1D numpy float32 array. Values typically in [-1, 1].

        Returns:
            Augmented waveform, 1D float32, clipped to [-1, 1].
        """
        x = np.asarray(x, dtype=np.float32)
        original_len = len(x)

        if self.rng.random() < self.p_lnl:
            x = self._apply_linear_conv_noise(x)
        if self.rng.random() < self.p_isd:
            x = self._apply_impulsive_noise(x)
        if self.rng.random() < self.p_ssi:
            x = self._apply_stationary_noise(x)
        if self.rng.random() < self.p_codec:
            x = self._apply_codec(x)

        # Safety: clip and ensure length preserved + correct dtype.
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
        if len(x) != original_len:
            # Defensive: should not happen with these ops, but guard anyway.
            x = x[:original_len] if len(x) > original_len else np.pad(
                x, (0, original_len - len(x))
            )
        return x

    # ──────────────────────────────────────────────────────────
    # Individual augmentations
    # ──────────────────────────────────────────────────────────

    def _apply_linear_conv_noise(self, x: np.ndarray) -> np.ndarray:
        """Linear convolutive noise: filter with random FIR taps, add at low gain.

        Models effects of unknown channel responses (microphones, room acoustics,
        codec frequency shaping) without explicit physical modeling.
        """
        n_taps = int(self.rng.integers(self.lnl_n_taps_range[0], self.lnl_n_taps_range[1] + 1))
        taps = self.rng.standard_normal(n_taps).astype(np.float32)
        taps /= np.abs(taps).sum() + 1e-8  # Normalize tap energy

        gain_db = self.rng.uniform(*self.lnl_gain_db_range)
        gain_lin = 10.0 ** (gain_db / 20.0)

        filtered = signal.lfilter(taps, [1.0], x).astype(np.float32)
        return x + gain_lin * filtered

    def _apply_impulsive_noise(self, x: np.ndarray) -> np.ndarray:
        """Sparse impulse spikes scattered randomly across the waveform.

        Models clicks, pops, and packet-loss artifacts from real-world channels.
        """
        n_impulses = int(self.rng.integers(self.isd_n_impulses_range[0], self.isd_n_impulses_range[1] + 1))
        if n_impulses == 0 or len(x) == 0:
            return x

        amplitude = self.rng.uniform(*self.isd_amplitude_range)
        positions = self.rng.choice(len(x), size=n_impulses, replace=False)
        signs = self.rng.choice([-1.0, 1.0], size=n_impulses)

        x = x.copy()
        x[positions] = x[positions] + signs.astype(np.float32) * np.float32(amplitude)
        return x

    def _apply_stationary_noise(self, x: np.ndarray) -> np.ndarray:
        """Broadband stationary Gaussian noise at random SNR.

        Models environmental background noise.
        """
        snr_db = self.rng.uniform(*self.ssi_snr_db_range)
        signal_power = float(np.mean(x.astype(np.float64) ** 2)) + 1e-10
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = (self.rng.standard_normal(len(x)) * np.sqrt(noise_power)).astype(np.float32)
        return x + noise

    def _apply_codec(self, x: np.ndarray) -> np.ndarray:
        """Random codec simulation via companding + 8-bit quantization."""
        codec = self.rng.choice(self.codec_choices)
        if codec == "mulaw":
            return self._mulaw_round_trip(x)
        if codec == "alaw":
            return self._alaw_round_trip(x)
        return x

    @staticmethod
    def _mulaw_round_trip(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
        """μ-law encode + 8-bit quantize + decode round-trip."""
        x = np.clip(x, -1.0, 1.0)
        # Encode
        encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        # Quantize to 8 bits (256 levels in [-1, 1])
        q = np.round(encoded * 127.0) / 127.0
        # Decode
        decoded = np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu))) / mu
        return decoded.astype(np.float32)

    @staticmethod
    def _alaw_round_trip(x: np.ndarray, A: float = 87.6) -> np.ndarray:
        """A-law encode + 8-bit quantize + decode round-trip."""
        x = np.clip(x, -1.0, 1.0)
        abs_x = np.abs(x)
        ln_A = np.log(A)

        # Encode
        encoded_abs = np.where(
            abs_x < 1.0 / A,
            (A * abs_x) / (1.0 + ln_A),
            (1.0 + np.log(np.maximum(A * abs_x, 1e-12))) / (1.0 + ln_A),
        )
        encoded = np.sign(x) * encoded_abs

        # Quantize to 8 bits
        q = np.round(encoded * 127.0) / 127.0

        # Decode
        abs_q = np.abs(q)
        decoded_abs = np.where(
            abs_q < 1.0 / (1.0 + ln_A),
            abs_q * (1.0 + ln_A) / A,
            np.exp(abs_q * (1.0 + ln_A) - 1.0) / A,
        )
        return (np.sign(q) * decoded_abs).astype(np.float32)


if __name__ == "__main__":
    # Quick smoke test
    rb = RawBoostAugment(seed=42)
    x = np.random.randn(64600).astype(np.float32) * 0.1
    print(f"Input: shape={x.shape} dtype={x.dtype} range=[{x.min():.3f}, {x.max():.3f}]")
    for trial in range(5):
        y = rb(x.copy())
        diff = np.mean((y - x) ** 2)
        print(f"Trial {trial}: shape={y.shape} dtype={y.dtype} "
              f"range=[{y.min():.3f}, {y.max():.3f}] mse_from_input={diff:.6f}")
