import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import librosa
import soundfile as sf
import torchaudio
import subprocess
import tempfile
from tqdm import tqdm
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
MAX_LEN = 4 * SR
BATCH_SIZE = 32
NUM_WORKERS = 4

# ============================================================
# AUDIO PROCESSING
# ============================================================

def finalize_waveform(waveform):
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    if len(waveform) > MAX_LEN:
        waveform = waveform[:MAX_LEN]
    else:
        waveform = np.pad(waveform, (0, MAX_LEN - len(waveform)))
    return torch.tensor(waveform, dtype=torch.float32)

def process_audio(path):
    # torchaudio
    try:
        wf, sr = torchaudio.load(path)
        wf = wf.mean(dim=0).numpy()
        if sr != SR:
            wf = librosa.resample(wf, orig_sr=sr, target_sr=SR)
        return finalize_waveform(wf)
    except:
        pass

    # soundfile
    try:
        wf, sr = sf.read(path)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        if sr != SR:
            wf = librosa.resample(wf, orig_sr=sr, target_sr=SR)
        return finalize_waveform(wf)
    except:
        pass

    # ffmpeg fallback
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SR), tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        wf, sr = librosa.load(tmp_path, sr=SR, mono=True)
        os.remove(tmp_path)
        return finalize_waveform(wf)
    except:
        return None

# ============================================================
# DATASET
# ============================================================

class ASVSpoofTestDataset(Dataset):
    def __init__(self, protocol_file, audio_dir):
        self.audio_dir = audio_dir
        self.data = []

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                file_id = parts[1]
                label_str = parts[-1].lower()
                if label_str not in ["spoof", "bonafide"]:
                    continue
                label = 1 if label_str == "bonafide" else 0
                attack_id = parts[-2]
                self.data.append((file_id, label, attack_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_id, label, attack_id = self.data[idx]
        path = os.path.join(self.audio_dir, file_id + ".flac")
        wf = process_audio(path)
        if wf is None:
            wf = torch.zeros(MAX_LEN).float()
        return wf, torch.tensor(label).long(), file_id, attack_id

# ============================================================
# MODEL
# ============================================================

class Wav2Vec2Deepfake(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden = self.wav2vec.config.hidden_size

        for p in self.wav2vec.parameters():
            p.requires_grad = False

        for layer in self.wav2vec.encoder.layers[-6:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        out = self.wav2vec(x).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled)

# ============================================================
# METRICS
# ============================================================

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    file_ids, attack_ids = [], []

    with torch.no_grad():
        for wf, labels, ids, attacks in tqdm(loader):
            wf = wf.to(DEVICE)
            out = model(wf)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            file_ids.extend(ids)
            attack_ids.extend(attacks)

    eer = compute_eer(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    return eer, cm, report, file_ids, attack_ids, y_true, y_pred, y_scores

# ============================================================
# MAIN
# ============================================================

def main(model_path, protocol, audio_dir):
    dataset = ASVSpoofTestDataset(protocol, audio_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    model = Wav2Vec2Deepfake().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    eer, cm, report, file_ids, attack_ids, y_true, y_pred, y_scores = evaluate(model, loader)

    print(f"\nEER: {eer:.2f}%")
    print(cm)
    print(report)

    df = pd.DataFrame({
        "file_id": file_ids,
        "attack_id": attack_ids,
        "true_label": y_true,
        "prediction": y_pred,
        "score": y_scores
    })
    df.to_csv("test_predictions.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio_dir", required=True)
    args = parser.parse_args()
    main(args.model, args.protocol, args.audio_dir)



