#%%
from __future__ import annotations
from typing import Tuple

from scipy import signal
import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import huggingface_hub
import matplotlib.pyplot as plt
import librosa



from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable
import torchaudio
from collections.abc import Sequence

#%%
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
#%%
from logmel_cnn_baseline.src.asv_baseline.data.asvspoof_dataset import ASVspoofItem, parse_la_protocol, _balanced_limit, ASVspoofLADataset, load_audio
# Import the Models for AASIST
from logmel_cnn_baseline.src.asv_baseline.evaluation.metrics import compute_binary_metrics, compute_eer
from deployment.aasist_lambda.vendor.aasist.models.AASIST import Model as AASISTModel 
from aasist.code.rawboost import RawBoostAugment
from aasist.code.codec_aug import CodecAugment
#%%
def initialize_aasist_model():

    # Model config (must match training)
    AASIST_CFG = {
        "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pre-trained AASIST model from Hugging Face Hub
    aasist_chkpt = huggingface_hub.hf_hub_download(
        repo_id="arnavjain321/aasist-v2-rawboost", 
        filename="best.pt", 
        repo_type="model"
    )
    
    # Load the state dictionary into the model
    state_dict = torch.load(aasist_chkpt, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model = AASISTModel(AASIST_CFG).to(device)
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def initialize_aasist_dataset(protocol_path: str, audio_root: str, file_ext: str = ".flac", limit: int | None = None, shuffle_seed: int | None = None, balanced_limit: bool = False) -> ASVspoofLADataset:
    # Parse the protocol file to get a list of ASVspoofItem instances
    items = parse_la_protocol(protocol_path, audio_root, file_ext, limit, shuffle_seed, balanced_limit)
    # Create an ASVspoofLADataset instance using the parsed items
    dataset = ASVspoofLADataset(items)
    
    return dataset

model=initialize_aasist_model()
protocol_path = r"C:\Users\eglha\Downloads\ASV_Spoof_Data\DS_10283_3336\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"
audio_root = r"C:\Users\eglha\Downloads\ASV_Spoof_Data\DS_10283_3336\LA\LA\ASVspoof2019_LA_eval\flac"

dataset=initialize_aasist_dataset(protocol_path=protocol_path, audio_root=audio_root, file_ext=".flac")

# %%
print(model)
# %%

# %%
def make_spectrogram(dataset_item, sr=20):
    waveform, label = dataset_item
    # Compute the spectrogram
    S = librosa.stft(waveform.numpy())
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram for label: {"bonafide" if label.item() == 0 else "spoof"}')
    plt.tight_layout()
    plt.show()
make_spectrogram(dataset[1])
# %%
dataset[1]
# %%
NB_SAMP = 64600

def robust_load(path):
    """Try torchaudio, then soundfile, then ffmpeg subprocess.
    Returns (waveform_1d_float32, ok_flag, err_str).
    """
    import subprocess
    err = None
    # 1) torchaudio (uses libsndfile/sox/torchcodec depending on install)
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).numpy().astype(np.float32)
        if sr != 16000:
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
            wav = wav_t.squeeze(0).numpy().astype(np.float32)
        return wav, True, None
    except Exception as e:
        err = f"torchaudio: {type(e).__name__}: {e}"
    # 2) soundfile (libsndfile direct)
    try:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(round(len(wav) * ratio))
            idx = np.linspace(0, len(wav) - 1, new_len).astype(int)
            wav = wav[idx]
        return wav.astype(np.float32), True, None
    except Exception as e:
        err = f"{err} | soundfile: {type(e).__name__}: {e}"
    # 3) ffmpeg subprocess (handles weird FLAC variants libsndfile chokes on)
    try:
        cmd = [
            "ffmpeg", "-loglevel", "quiet", "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", "16000", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        wav = np.frombuffer(result.stdout, dtype=np.float32).copy()
        if len(wav) == 0:
            raise RuntimeError("ffmpeg returned empty stream")
        return wav, True, None
    except Exception as e:
        err = f"{err} | ffmpeg: {type(e).__name__}: {e}"
    return np.zeros(NB_SAMP, dtype=np.float32), False, err

def fit_length(wav, n_target=NB_SAMP):
    n = len(wav)
    if n == n_target:
        return wav
    if n > n_target:
        start = (n - n_target) // 2
        return wav[start:start + n_target]
    # repeat-pad
    repeats = (n_target + n - 1) // n
    return np.tile(wav, repeats)[:n_target]


class RobustAASISTDataset(Dataset):
    def __init__(self, items):
        self.items = items  # list of ASVspoofItem from parse_la_protocol

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav, ok, err = robust_load(item.path)
        wav = fit_length(wav)
        return {
            "waveform": torch.from_numpy(wav.astype(np.float32)),
            "label": torch.tensor(item.label, dtype=torch.long),
            "utterance_id": item.utterance_id,
            "ok": ok,
            "err": err if err else "",
        }

def initialize_aasist_dataset(protocol_path, audio_root, file_ext=".flac", limit=None):
    items = parse_la_protocol(protocol_path, audio_root, file_ext, limit, shuffle_seed=None, balanced_limit=False)
    return RobustAASISTDataset(items)

dataset = initialize_aasist_dataset(protocol_path, audio_root, file_ext=".flac", limit=None)
# %%
@torch.no_grad()
def predict_scores(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        x = batch["waveform"].to(device)
        _, logits = model(x)
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        for utt_id, lbl, score in zip(batch["utterance_id"], batch["label"], scores):
            rows.append({"utterance_id": utt_id, "label": lbl.item(), "score": score})
    return rows

predict_scores(model, DataLoader(dataset, batch_size=16), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# %%
