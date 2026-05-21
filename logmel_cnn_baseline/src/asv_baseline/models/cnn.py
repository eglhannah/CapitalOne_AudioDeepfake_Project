from __future__ import annotations

from collections.abc import Sequence

from torch import nn


class LogMelCNN(nn.Module):
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 128), dropout: float = 0.25) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 1),
        )

    def forward(self, features):
        x = self.encoder(features)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)
