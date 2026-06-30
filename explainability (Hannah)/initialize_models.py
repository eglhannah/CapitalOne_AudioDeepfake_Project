#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import torch.nn.functional as F

import matplotlib.pyplot as plt
import plotly.express as px

import huggingface_hub
from transformers import Wav2Vec2Model
import os
from dotenv import load_dotenv
import random
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from collections import OrderedDict
import soundfile as sf
import librosa

from models.AASIST import Model as AASISTModel
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

    model = model_name.to(device)
    # Download the trained checkpoint from Hugging Face Hub
    wav2vec_name='best_model.pth'
    trained_checkpoint=hf_hub_download(
    repo_id='rde6mn/no_aug_w2v_4s',
    filename=wav2vec_name)

    # Load the trained checkpoint to intialize the model to the trained weights
    checkpoint = torch.load(trained_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

#%%
def initialize_aasist_model(model_name=, device=DEVICE):
    model = 
    ckpt = torch.load(map_location=device, weights_only=False)
    model = AASISTModel(AASIST_CFG).to(device)
    model.load_state_dict(ckpt["model"])

    ds = ASVspoofLADataset(
        items, sample_rate=16000, duration_sec=64600 / 16000, training=False
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return model, loader



#%%
def initialize_logmel_model(model_name=, device=DEVICE):
    model = 
    model.eval()
    return model