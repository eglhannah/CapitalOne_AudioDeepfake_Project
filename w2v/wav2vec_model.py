import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    Wav2Vec2Model
)

from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    classification_report
)

from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt



class Wav2Vec2Deepfake(nn.Module):

    def __init__(self):

        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        hidden_size = (
            self.wav2vec.config.hidden_size
        )

        # ============================================
        # FREEZE MOST OF MODEL
        # ============================================

        for param in self.wav2vec.parameters():

            param.requires_grad = False

        # ============================================
        # UNFREEZE LAST 2 ENCODER LAYERS
        # ============================================

        for layer in self.wav2vec.encoder.layers[-2:]:

            for param in layer.parameters():

                param.requires_grad = True

        # ============================================
        # CLASSIFIER
        # ============================================

        self.classifier = nn.Sequential(

            nn.Linear(hidden_size, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 2)
        )

    def forward(self, x):

        outputs = self.wav2vec(
            x
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        pooled = hidden_states.mean(
            dim=1
        )

        logits = self.classifier(
            pooled
        )

        return logits