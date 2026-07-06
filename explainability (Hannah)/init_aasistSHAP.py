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



from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable
import torchaudio
from collections.abc import Sequence

# Import Dataset for AASIST Model
from logmel_cnn_baseline.src.asv_baseline.data.asvspoof_dataset import ASVspoofItem, parse_la_protocol, _balanced_limit, ASVspoofLADataset, load_audio
# Import the Models for AASIST
from logmel_cnn_baseline.src.asv_baseline.data.asvspoof_dataset import ASVspoofLADataset, parse_la_protocol
from logmel_cnn_baseline.src.asv_baseline.evaluation.metrics import compute_binary_metrics
from deployment.aasist_lambda.vendor.aasist.models.AASIST import GraphAttentionLayer as AASISTModel 
from aasist.code.rawboost import RawBoostAugment
from aasist.code.codec_aug import CodecAugment

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

initialize_aasist_model()
protocol_path = r"C:\Users\eglha\Downloads\ASV_Spoof_Data\DS_10283_3336\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
audio_root = r"C:\Users\eglha\Downloads\ASV_Spoof_Data\DS_10283_3336\LA\LA\ASVspoof2019_LA_eval\flac"

initialize_aasist_dataset(protocol_path=protocol_path, audio_root=audio_root, file_ext=".flac")
