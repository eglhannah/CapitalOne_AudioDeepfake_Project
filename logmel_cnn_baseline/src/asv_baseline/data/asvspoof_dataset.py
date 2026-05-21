from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ASVspoofItem:
    utterance_id: str
    path: Path
    label: int
    label_name: str
    speaker_id: str | None = None
    attack_id: str | None = None


def parse_la_protocol(
    protocol_path: str | Path,
    audio_root: str | Path,
    file_ext: str = ".flac",
    limit: int | None = None,
    shuffle_seed: int | None = None,
    balanced_limit: bool = False,
) -> list[ASVspoofItem]:
    """Parse an ASVspoof 2019 LA protocol file.

    Expected rows contain whitespace-separated fields where the second token is
    the utterance id and the last token is either "bonafide" or "spoof".
    This covers the ASVspoof 2019 LA train/dev CM protocol format.
    """
    protocol_path = Path(protocol_path)
    audio_root = Path(audio_root)
    items: list[ASVspoofItem] = []

    with protocol_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed protocol row at {protocol_path}:{line_no}: {line!r}")

            label_name = parts[-1].lower()
            if label_name not in {"bonafide", "spoof"}:
                raise ValueError(
                    f"Expected last token to be bonafide/spoof at "
                    f"{protocol_path}:{line_no}, got {parts[-1]!r}"
                )

            utterance_id = parts[1]
            audio_name = utterance_id if Path(utterance_id).suffix else f"{utterance_id}{file_ext}"
            items.append(
                ASVspoofItem(
                    utterance_id=utterance_id,
                    path=audio_root / audio_name,
                    label=0 if label_name == "bonafide" else 1,
                    label_name=label_name,
                    speaker_id=parts[0] if parts else None,
                    attack_id=parts[3] if len(parts) > 3 else None,
                )
            )

    if not items:
        raise ValueError(f"No usable rows found in protocol file: {protocol_path}")

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(items)

    if limit is not None:
        if balanced_limit:
            items = _balanced_limit(items, limit)
        else:
            items = items[:limit]

    return items


def _balanced_limit(items: list[ASVspoofItem], limit: int) -> list[ASVspoofItem]:
    by_label = {
        0: [item for item in items if item.label == 0],
        1: [item for item in items if item.label == 1],
    }
    per_class = limit // 2
    selected = by_label[0][:per_class] + by_label[1][:per_class]

    remainder = limit - len(selected)
    if remainder > 0:
        used_ids = {id(item) for item in selected}
        extras = [item for item in items if id(item) not in used_ids]
        selected.extend(extras[:remainder])

    return selected


class ASVspoofLADataset(Dataset):
    def __init__(
        self,
        items: Iterable[ASVspoofItem],
        sample_rate: int = 16_000,
        duration_sec: float = 4.0,
        training: bool = False,
    ) -> None:
        self.items = list(items)
        self.sample_rate = sample_rate
        self.duration_samples = int(round(sample_rate * duration_sec))
        self.training = training

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.items[index]
        waveform, sr = load_audio(item.path)
        waveform = self._to_mono(waveform)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = self._fit_length(waveform)

        return {
            "waveform": waveform.squeeze(0),
            "label": torch.tensor(item.label, dtype=torch.float32),
            "utterance_id": item.utterance_id,
            "path": str(item.path),
        }

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(0) == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _fit_length(self, waveform: torch.Tensor) -> torch.Tensor:
        current = waveform.size(-1)
        target = self.duration_samples

        if current == target:
            return waveform

        if current > target:
            if self.training:
                start = torch.randint(0, current - target + 1, size=(1,)).item()
            else:
                start = (current - target) // 2
            return waveform[:, start : start + target]

        repeats = (target + current - 1) // current
        padded = waveform.repeat(1, repeats)
        return padded[:, :target]


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio while avoiding a hard dependency on TorchCodec.

    Some recent torchaudio installs route decoding through TorchCodec. Many HPC
    environments have torchaudio but not torchcodec, while still supporting FLAC
    through soundfile/libsndfile. This fallback keeps the dataset portable.
    """
    path = Path(path)
    try:
        return torchaudio.load(str(path))
    except ImportError as exc:
        if "TorchCodec" not in str(exc) and "torchcodec" not in str(exc):
            raise

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "torchaudio requires TorchCodec in this environment and the soundfile "
            "fallback is not installed. Install either torchcodec or soundfile."
        ) from exc

    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(np.asarray(audio).T)
    return waveform, int(sample_rate)
