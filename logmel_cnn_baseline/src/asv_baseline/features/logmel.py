from __future__ import annotations

import torch
from torch import nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length_ms: int = 25,
        hop_length_ms: int = 10,
        f_min: float = 20.0,
        f_max: float | None = 7600.0,
    ) -> None:
        super().__init__()
        win_length = int(round(sample_rate * win_length_ms / 1000))
        hop_length = int(round(sample_rate * hop_length_ms / 1000))
        if n_fft < win_length:
            raise ValueError(f"n_fft must be >= win_length; got n_fft={n_fft}, win_length={win_length}")

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
            normalized=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        features = self.to_db(self.mel(waveform))
        mean = features.mean(dim=(-2, -1), keepdim=True)
        std = features.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (features - mean) / std
