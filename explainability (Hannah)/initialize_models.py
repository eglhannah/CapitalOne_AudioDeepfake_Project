#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt
import plotly.express as px

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import huggingface_hub
from transformers import Wav2Vec2Model
import os
from dotenv import load_dotenv
import random
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader, random_split, dataclass
from collections import OrderedDict
import soundfile as sf
import librosa

# from models.AASIST import Model as AASISTModel
from transformers import Wav2Vec2Model

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from huggingface_hub import hf_hub_download


# %%
aasistV1_model_path=r'https://huggingface.co/arnavjain321/aasist-v1-baseline'
aasistV1_name='best.pt'
aasistV1_config_name='congif.json'


aasistV2_model_path=r'https://huggingface.co/arnavjain321/aasist-v2-rawboost'
aasistV2_name='best.pt'
aasistV2_config_name='congif.json'

wav2vec_model_path=r'https://huggingface.co/rde6mn/no_aug_w2v_4s'
wav2vec_name='best_model.pth'

logmel_model_path=r'https://huggingface.co/chasecha/logmel_cnn_baseline'
log_mel_name='logmel_cnn_baseline.pt'
#%%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%%
# Wav2Vec2 Model
class Wav2Vec2Deepfake(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True)
        hidden_size = self.wav2vec.config.hidden_size
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        for layer in self.wav2vec.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, x):
        outputs = self.wav2vec(x)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
    

class ASVSpoofDataset(Dataset, dataset_type="2019"):
    def __init__(
        self,
        protocol_file,
        dataset_type="2019"):
        self.data = []
        with open(protocol_file, "r") as f:
            lines = f.readlines()
        if dataset_type == "2019":
            for line in lines:
                parts = line.strip().split()
                file_id = parts[1]
                label = parts[-1]
                label = (
                    1 if label == "bonafide"
                    else 0)
                self.data.append(
                    (file_id, label))
        else:
            for line in lines:
                parts = line.strip().split()
                # Expected format:
                # speaker  file_id  codec  source  attack  label  ...
                # Example:
                # LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim ...
                if len(parts) < 6:
                    continue
                file_id = parts[1]      # DF_E_2000011
                label_str = parts[5]    # spoof / bonafide
                if label_str not in ["spoof", "bonafide"]:
                    continue
                label = 1 if label_str == "bonafide" else 0
                self.data.append((file_id, label))


    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        file_id, label = self.data[idx]
        cache_path = os.path.join(CACHE_DIR, file_id + ".pt")
        if not os.path.exists(cache_path):
            return self.__getitem__(random.randint(0, len(self.data)-1))
        waveform = torch.load(cache_path)
        return waveform, torch.tensor(label).long()
    

def initialize_wav2vec2(model_name=Wav2Vec2Deepfake(), device=DEVICE):
    SR = 16000
    MAX_LEN = 4 * SR
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 1e-5
    NUM_WORKERS = 4

    model = Wav2Vec2Deepfake().to(
        DEVICE
    )
    trained_checkpoint= hf_hub_download(repo_id='rde6mn/no_aug_w2v_4s', filename='best_model.pth')
    checkpoint = torch.load(trained_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

#%%

#------------------------------------------------------------------------------

## Dataset for AASIST Model ##

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


def initialize_aasist_model(model_name=AASISTModel(), device=DEVICE):
    model = model_name.to(device)
    checkpoint = hf_hub_download(
        repo_id='arnavjain321/aasist-v2-rawboost',
        filename='best.pt',
        config_filename='config.json'
    )
    checkpoint = torch.load(map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    batch_size=24
    ds = ASVspoofLADataset(
        items, sample_rate=16000, duration_sec=64600 / 16000, training=False
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    return model, loader



#%%
