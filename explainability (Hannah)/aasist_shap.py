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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from aasist.simple_aasist import load_aasist_v3, predict
import soundfile as sf

from init_aasistSHAP import DEVICE, model, background, predictions, audio_dir, audio_files, protocols_df, aasist_wrapper, samples

def extract_audiofeatures(predictions):
    avg_features = []
    for item in predictions.itertuples():
        filename = item.file
        audio_path = Path(audio_dir / filename)
        waveform, sr = sf.read(audio_path)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        y = waveform.numpy().squeeze()

        # Onset Envelope for Librosa functions
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Spectral Flatness
        flatness=librosa.feature.spectral_flatness(y=y)

        # Mel-Frequency Cepstral Coefficients (MFCCs)
        mfccs=librosa.feature.mfcc(y=y, sr=sr)

        # Harmonic-Percussive Source Separation (HPSS)
        hpss=np.array(librosa.effects.hpss(y))

        groundtruth = item.groundtruth


        
        # Append to avg_features
        avg_features.append({
            "file": item.file,
            "spoof_prob": item.spoof_prob,
            "logits": item.logits,
            "groundtruth": groundtruth,
            "onset_env": onset_env.tolist(),
            "spectral_flatness": flatness.tolist(),
            "mfccs": mfccs.tolist(),
            "hpss": hpss.tolist()
        })

    return avg_features

extracted_features = extract_audiofeatures(predictions)
print(extracted_features[0])

extracted_features_df = pd.DataFrame(extracted_features)
extracted_features_df.head()

import numpy as np

def summarize_audio_features(df):

    rows = []

    for _, row in df.iterrows():
        feature_row = {
            "file": row["file"],
            "groundtruth": row["groundtruth"],
            "spoof_prob": row["spoof_prob"]
        }

        # MFCC summary
        mfcc = np.array(row["mfccs"])
        feature_row.update({
            f"mfcc_{i}_mean": mfcc[i].mean()
            for i in range(mfcc.shape[0])
        })
        feature_row.update({
            f"mfcc_{i}_std": mfcc[i].std()
            for i in range(mfcc.shape[0])
        })

        # Spectral flatness
        flatness = np.array(row["spectral_flatness"])
        feature_row["flatness_mean"] = flatness.mean()
        feature_row["flatness_std"] = flatness.std()

        # Onset envelope
        onset = np.array(row["onset_env"])
        feature_row["onset_mean"] = onset.mean()
        feature_row["onset_std"] = onset.std()

        # HPSS
        hpss = np.array(row["hpss"])
        feature_row["harmonic_mean"] = hpss[0].mean()
        feature_row["percussive_mean"] = hpss[1].mean()

        rows.append(feature_row)

    return pd.DataFrame(rows)


feature_summary = summarize_audio_features(extracted_features_df)
feature_summary.head()
feature_summary["groundtruth"].value_counts()



sns.boxplot(
    data=feature_summary,
    x="groundtruth",
    y="mfcc_0_mean"
)

plt.title("MFCC 0 Mean by Class")
plt.show()

def subplot_features_byclass(df):
    features_to_plot = [
        "mfcc_0_mean", "mfcc_1_mean", "mfcc_2_mean",
        "flatness_mean", "onset_mean",
        "harmonic_mean", "percussive_mean"
    ]

    fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(10, 20))

    for i, feature in enumerate(features_to_plot):
        sns.boxplot(
            data=df,
            x="groundtruth",
            y=feature,
            ax=axes[i]
        )
        axes[i].set_title(f"{feature} by Class")

    plt.tight_layout()
    plt.show()

subplot_features_byclass(feature_summary)
class_summary=feature_summary[['groundtruth', 'mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'flatness_mean', 'onset_mean', 'harmonic_mean', 'percussive_mean']].groupby('groundtruth').mean()
class_summary



X = feature_summary.drop(
    columns=["file", "groundtruth"]
)

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)

components = pca.fit_transform(X_scaled)

feature_summary["PC1"] = components[:,0]
feature_summary["PC2"] = components[:,1]
sns.scatterplot(
    data=feature_summary,
    x="PC1",
    y="PC2",
    hue="groundtruth",
    alpha=0.7
)

plt.show()