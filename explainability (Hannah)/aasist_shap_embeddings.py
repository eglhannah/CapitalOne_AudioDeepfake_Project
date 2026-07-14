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
import sys
from pathlib import Path
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import shap
import torch



sys.path.append(str(Path(__file__).resolve().parents[1]))
from aasist.simple_aasist import load_aasist_v3, predict
import soundfile as sf

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading AASIST v3 from hf repo on {DEVICE}...")
model = load_aasist_v3()
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded. Params: {n_params:,}")

#%%
audio_dir=Path(r"I:\My Drive\ASVSpoof_Data\unzipped2019\LA\LA\ASVspoof2019_LA_dev\flac")
predictions=[]
audio_files = list(audio_dir.glob("*.flac"))
random.shuffle(audio_files)
audio_files=audio_files[:1000]  # Limit to 1000 files for testing
for audio in tqdm(audio_files, total=len(audio_files)):
    waveform, sr = sf.read(audio)
    waveform = torch.from_numpy(waveform).float().unsqueeze(0)
    out = predict(model, waveform)
    predictions.append({
        "file": audio.name,
        "spoof_prob": out["spoof_prob"].item(),
        "logits": out["logits"].squeeze(0).tolist(),
        "embedding": out["embedding"].squeeze(0).tolist()
        })

# %%
predictions = pd.DataFrame(predictions)
predictions.to_csv("aasist_v3_predictions.csv", index=False)
# %%


#%%
def parse_protocols_LADEV(protocol_file: str) -> dict:

    protocols = {}
    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                filename = parts[1]
                label = parts[4]

                protocols[filename] = label
    return protocols

protocols = parse_protocols_LADEV(r"I:\My Drive\ASVSpoof_Data\unzipped2019\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt")
protocols_df = pd.DataFrame(list(protocols.items()), columns=['filename', 'groundtruth'])
protocols_df["groundtruth"].value_counts()

#%%
def make_spectrogram(dataset_item):
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
make_spectrogram((torch.from_numpy(audio).float(), torch.tensor(0)))  # Assuming label is 0 for bonafide
# %%
# Join Real Protocols label to Predictions Table
predictions["filename"] = predictions["file"].str.replace(".flac", "")
predictions["groundtruth"] = predictions["filename"].map(protocols)
predictions.head()
# %%
predictions['groundtruth'].value_counts()

# %%
def shap_model(audio):
    """Wrapper function to use AASIST model with SHAP."""
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # Add batch dimension
    out = predict(model, audio_tensor)
    return out["spoof_prob"].detach().cpu().numpy()  # Return numpy array for SHAP

def fix_length(waveform, target_length=TARGET_LENGTH):
    length = waveform.shape[0]

    if length < target_length:
        repeats = (target_length // length) + 1
        waveform = waveform.repeat(repeats)[:target_length]

    elif length > target_length:
        waveform = waveform[:target_length]

    return waveform

def create_tensors(audio_files):
    """Create a tensor of audio samples."""
    audio_tensors = []
    for audio in tqdm(audio_files):
        waveform, sr = sf.read(audio)
        waveform = torch.from_numpy(waveform).float()
        # Convert stereo to mono
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)
        # Ensure AASIST input size
        waveform = fix_length(waveform)
        audio_tensors.append(waveform)

    return torch.stack(audio_tensors)

audio_tensors = create_tensors(audio_files)
print("Audio tensor shape:", audio_tensors.shape)


background=audio_tensors[:100].to(DEVICE)

#%%

TARGET_LENGTH = 64600

def fix_length(waveform, target_length=TARGET_LENGTH):
    length = waveform.shape[0]

    if length < target_length:
        # repeat pad
        repeats = (target_length // length) + 1
        waveform = waveform.repeat(repeats)[:target_length]

    elif length > target_length:
        # crop
        waveform = waveform[:target_length]

    return waveform


embeddings = []
audio_tensors = []

for audio in tqdm(audio_files):

    waveform, sr = sf.read(audio)

    waveform = torch.from_numpy(waveform).float()

    # convert stereo to mono if needed
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    waveform = fix_length(waveform, target_length=64600)

    out = predict(model, waveform.unsqueeze(0))

    embeddings.append(out["embedding"].squeeze(0))
    audio_tensors.append(waveform)

embeddings = torch.stack(embeddings)
audio_tensors = torch.stack(audio_tensors)


print(embeddings.shape)
print(audio_tensors.shape)

# %%


class AASISTHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.out_layer = model.out_layer

    def forward(self, embedding):
        logits = self.out_layer(embedding)
        return logits[:,1].unsqueeze(1)
    

head = AASISTHead(model).to(DEVICE)
head.eval()

background = embeddings[:100].to(DEVICE)
samples = embeddings[100:110].to(DEVICE)

explainer = shap.DeepExplainer(
    head,
    background
)

shap_values = explainer.shap_values(
    samples,
    check_additivity=False)
shap_values = shap_values.squeeze(-1)

print(shap_values.shape)
# %%

mean_importance = np.abs(shap_values).mean(axis=0)

top_features = np.argsort(mean_importance)[::-1][:20]

for idx in top_features:
    print(f"Embedding dimension {idx}: {mean_importance[idx]:.5f}")
# %%
sample_idx = 0

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value[0],
        data=samples[sample_idx].detach().cpu().numpy()
    )
)
# %%


def explain_aasist(idx, shap_values, embeddings, samples, audio_files, protocol):
    
    # Audio/model information
    waveform = audio_tensors[idx].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = predict(model, waveform)

    spoof_prob = out["spoof_prob"].item()
    logits = out["logits"].squeeze(0)

    pred = int(torch.argmax(logits))
    pred_label = "spoof" if pred == 1 else "bonafide"

    # Filename
    file_name = audio_files[idx].stem

    # Find true label
    protocol_row = protocol[protocol['filename'] == file_name]

    if len(protocol_row) > 0:
        label = protocol_row["groundtruth"].values[0]
        label_num = 1 if label == "spoof" else 0
    else:
        label = "unknown"
        label_num = None

    print(f"File Name: {file_name}")
    print("-----------------------------")
    print(f"True Label: {label_num}: {label}")
    print(f"Predicted Label: {pred}: {pred_label}")
    print(f"Spoof Probability: {spoof_prob:.4f}")


    # SHAP explanation
    sample_shap = shap_values[idx]

    plt.figure(figsize=(12,4))
    plt.bar(
        range(len(sample_shap)),
        sample_shap
    )
    plt.xlabel("AASIST Embedding Dimension")
    plt.ylabel("SHAP Value")
    plt.title("AASIST Embedding Feature Contributions")
    plt.axhline(0)
    plt.show()

    # Waterfall plot
    shap.plots.waterfall(
        shap.Explanation(
            values=sample_shap,
            base_values=explainer.expected_value[0],
            data=embeddings[idx].cpu().numpy()
        )
    )

    
explain_aasist(idx=5, shap_values=shap_values, embeddings=embeddings, samples=samples, audio_files=audio_files, protocol=protocols_df)

# %%
print(protocols_df.columns)
print(protocols_df.head())

# %%
