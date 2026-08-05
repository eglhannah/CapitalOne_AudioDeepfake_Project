"""
interpret_aasist_shap.py
========================
Explains *why* AASIST v3 classifies audio as spoof or bonafide by computing
SHAP at three complementary levels:

  1. Embedding-level  – which dimensions of the 128-d pooled embedding push
     the decision toward spoof vs bonafide.
  2. Temporal-level   – which time-frames of the raw 64 600-sample waveform
     matter most (via a GREEDY audio wrapper).
  3. Aggregate-level  – across many samples, which embedding dimensions are
     consistently the strongest indicators, and how do correct vs incorrect
     predictions differ?

Outputs are saved to a ./interpretation_results/ directory (plots + CSV).

Usage:
    python interpret_aasist_shap.py --data_dir <path_to_flac> \
                                    --protocol <path_to_trl.txt> \
                                    --n_background 100 \
                                    --n_samples 50 \
                                    --device cuda
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torchaudio

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from aasist.simple_aasist import load_aasist_v3, predict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_LENGTH = 64600
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ===================================================================
# Audio helpers
# ===================================================================

def fix_length(waveform: torch.Tensor, target: int = TARGET_LENGTH) -> torch.Tensor:
    n = waveform.shape[0]
    if n < target:
        reps = (target // n) + 1
        waveform = waveform.repeat(reps)[:target]
    elif n > target:
        waveform = waveform[:target]
    return waveform


def load_audio(path: Path) -> torch.Tensor:
    wav, _ = sf.read(path)
    wav = torch.from_numpy(wav).float()
    if wav.ndim > 1:
        wav = wav.mean(dim=1)
    return fix_length(wav)


def parse_protocol(protocol_file: str) -> pd.DataFrame:
    rows = {}
    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                rows[parts[1]] = parts[4]
    return pd.DataFrame(list(rows.items()), columns=["filename", "groundtruth"])


# ===================================================================
# SHAP wrapper classes
# ===================================================================

class AASISTEmbeddingHead(nn.Module):
    """Expose *only* the classification head so SHAP operates on the
    128-d pooled embedding that the GAT pipeline produces."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.out_layer = model.out_layer

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        logits = self.out_layer(embedding)
        return logits[:, 1].unsqueeze(1)  # spoof logit


class AASISTAudioWrapper(nn.Module):
    """Full forward pass exposed as a Module so DeepExplainer can
    trace gradients through the raw waveform -> spoof logit."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        for m in self.model.modules():
            if isinstance(m, nn.SELU):
                m.inplace = False

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        _, logits = self.model(audio)
        return logits[:, 1].unsqueeze(1)


class SpectrogramSurrogate(nn.Module):
    """Small MLP that maps a flattened mel-spectrogram to a spoof
    probability.  Used as a surrogate for AASIST so SHAP can operate
    in the spectrogram space without hitting in-place-op issues."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================================================================
# Spectrogram helpers
# ===================================================================

MEL_N_MELS = 40
MEL_N_FFT = 1024
MEL_HOP_LENGTH = 512
SR = 16000


def compute_mel_spectrograms(
    audio_tensors: torch.Tensor,
    n_mels: int = MEL_N_MELS,
    n_fft: int = MEL_N_FFT,
    hop_length: int = MEL_HOP_LENGTH,
) -> torch.Tensor:
    """Compute log-mel spectrograms for a batch of audio waveforms.

    Parameters
    ----------
    audio_tensors : (n_samples, TARGET_LENGTH)

    Returns
    -------
    specs : (n_samples, n_mels, time_frames)  –  log-mel spectrograms
    """
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    specs = []
    for wav in audio_tensors:
        s = mel_spec(wav.unsqueeze(0))       # (1, n_mels, T)
        s = torch.log(s.clamp(min=1e-9))     # log-mel
        specs.append(s.squeeze(0))
    return torch.stack(specs)


def train_surrogate(
    surrogate: SpectrogramSurrogate,
    specs_flat: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """Train the surrogate MLP to predict AASIST spoof probabilities."""
    surrogate = surrogate.to(device)
    surrogate.train()
    optim = torch.optim.Adam(surrogate.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = specs_flat.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x = specs_flat[idx].to(device)
            y = targets[idx].to(device)
            pred = surrogate(x)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(idx)
    return epoch_loss / n


# ===================================================================
# Core analysis functions
# ===================================================================

def compute_occlusion_sensitivity(
    model: nn.Module,
    audio_tensors: torch.Tensor,
    n_windows: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Occlusion sensitivity over the raw waveform.

    For each audio sample, divides the 64 600-sample waveform into
    ``n_windows`` equal segments.  Each segment is zeroed out in turn
    and the change in spoof probability is recorded.  A large positive
    delta means that segment was important for the spoof decision; a
    large negative delta means it was important for the bonafide decision.

    Returns
    -------
    importance : np.ndarray, shape (n_samples, n_windows)
        Delta spoof probability when each window is occluded.
    baseline_probs : np.ndarray, shape (n_samples,)
        Original spoof probability before any occlusion.
    """
    model.eval()
    window_len = audio_tensors.shape[1] // n_windows
    n_samples = audio_tensors.shape[0]
    importance = np.zeros((n_samples, n_windows))
    baseline_probs = np.zeros(n_samples)

    for i in tqdm(range(n_samples), desc="Occlusion sensitivity"):
        wav = audio_tensors[i].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            base_out = predict(model, wav)
        base_prob = base_out["spoof_prob"].item()
        baseline_probs[i] = base_prob

        for w in range(n_windows):
            occluded = audio_tensors[i].clone()
            start = w * window_len
            end = start + window_len
            occluded[start:end] = 0.0
            with torch.no_grad():
                occ_out = predict(model, occluded.unsqueeze(0).to(DEVICE))
            occ_prob = occ_out["spoof_prob"].item()
            importance[i, w] = base_prob - occ_prob

    return importance, baseline_probs


def compute_spectrogram_shap(
    model: nn.Module,
    audio_tensors: torch.Tensor,
    n_mels: int = MEL_N_MELS,
    n_fft: int = MEL_N_FFT,
    hop_length: int = MEL_HOP_LENGTH,
    surrogate_epochs: int = 200,
    background_n: int = 100,
    n_samples: int = 50,
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """Train a surrogate MLP on mel spectrograms, then explain it with SHAP.

    Returns
    -------
    shap_values : (n_samples, n_mels, time_frames)
    spectrograms : (n_all, n_mels, time_frames)
    expected_value : float
    n_mels_out, n_frames_out : int, int
    """
    # 1. Compute mel spectrograms
    print("  Computing mel spectrograms...")
    specs = compute_mel_spectrograms(audio_tensors, n_mels, n_fft, hop_length)
    n_all, n_mels_out, n_frames = specs.shape
    n_features = n_mels_out * n_frames
    print(f"  Spectrogram shape: {specs.shape}  ({n_features} features)")

    specs_flat = specs.reshape(n_all, -1)

    # 2. Get AASIST predictions as training targets
    print("  Getting AASIST predictions for surrogate training...")
    targets = []
    for i in tqdm(range(n_all), desc="  AASIST inference"):
        wav = audio_tensors[i].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = predict(model, wav)
        targets.append(out["spoof_prob"].item())
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    # 3. Train surrogate
    print(f"  Training surrogate MLP ({surrogate_epochs} epochs)...")
    surrogate = SpectrogramSurrogate(n_features)
    final_loss = train_surrogate(
        surrogate, specs_flat, targets,
        epochs=surrogate_epochs, device=str(DEVICE),
    )
    print(f"  Surrogate final MSE: {final_loss:.6f}")

    # Evaluate surrogate accuracy
    surrogate.eval()
    with torch.no_grad():
        pred_all = surrogate(specs_flat.to(DEVICE)).cpu().squeeze()
    corr = np.corrcoef(pred_all.numpy(), targets.squeeze().numpy())[0, 1]
    print(f"  Surrogate-AASIST correlation: {corr:.4f}")

    # 4. SHAP on the surrogate
    print("  Running SHAP DeepExplainer on surrogate...")
    bg = specs_flat[:background_n].to(DEVICE)
    samples = specs_flat[background_n:background_n + n_samples].to(DEVICE)
    explainer = shap.DeepExplainer(surrogate, bg)
    sv = explainer.shap_values(samples, check_additivity=False)
    sv = np.asarray(sv)
    # Shape may be (n_samples, n_features, 1) or (n_samples, n_features)
    if sv.ndim == 3:
        sv = sv.squeeze(-1)
    sv = sv.reshape(n_samples, n_mels_out, n_frames)

    return sv, specs.numpy(), float(explainer.expected_value[0]), n_mels_out, n_frames

def compute_embedding_shap(
    model: nn.Module,
    embeddings: torch.Tensor,
    background_n: int = 100,
    n_samples: int = 50,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, float]:
    """SHAP DeepExplainer over embedding dimensions.

    Returns (shap_values, samples, background, expected_value).
    """
    head = AASISTEmbeddingHead(model).to(DEVICE).eval()
    bg = embeddings[:background_n].to(DEVICE)
    samples = embeddings[background_n : background_n + n_samples].to(DEVICE)

    explainer = shap.DeepExplainer(head, bg)
    sv = explainer.shap_values(samples, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 4:
        sv = sv.squeeze(-1)  # (n_samples, n_features)
    if sv.ndim == 3:
        sv = sv.squeeze(-1)  # (n_samples, n_features)

    return sv, samples, bg, float(explainer.expected_value[0])


def compute_audio_shap(
    model: nn.Module,
    audio_tensors: torch.Tensor,
    background_n: int = 30,
    n_samples: int = 10,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, float]:
    """SHAP over raw waveform samples.

    Tries DeepExplainer first, falls back to GradientExplainer, then
    KernelExplainer.  Because the audio tensor is 64 600 dims this is
    *expensive*; we keep sample counts deliberately small.
    """
    wrapper = AASISTAudioWrapper(model).to(DEVICE).eval()
    bg = audio_tensors[:background_n].to(DEVICE)
    samples = audio_tensors[background_n : background_n + n_samples].to(DEVICE)

    explainer = None
    method = None

    # --- Try DeepExplainer ---
    try:
        print("  Trying DeepExplainer...")
        explainer = shap.DeepExplainer(wrapper, bg)
        sv = explainer.shap_values(samples, check_additivity=False)
        method = "DeepExplainer"
        print("  DeepExplainer succeeded.")
    except Exception as e:
        print(f"  DeepExplainer failed: {e}")
        explainer = None

    # --- Fallback: GradientExplainer ---
    if explainer is None:
        try:
            print("  Trying GradientExplainer...")
            explainer = shap.GradientExplainer(wrapper, bg)
            sv = explainer.shap_values(samples)
            method = "GradientExplainer"
            print("  GradientExplainer succeeded.")
        except Exception as e:
            print(f"  GradientExplainer failed: {e}")
            explainer = None

    # --- Fallback: KernelExplainer ---
    if explainer is None:
        BLOCK = 6460
        n_blocks = audio_tensors.shape[1] // BLOCK
        print(f"  Falling back to KernelExplainer on {n_blocks} blocks of {BLOCK} samples...")

        def block_model_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32, device="cpu")
            x_upsampled = x_tensor.repeat_interleave(BLOCK, dim=1)
            with torch.no_grad():
                wrapper_cpu = wrapper.cpu()
                out = wrapper_cpu(x_upsampled)
                return out.cpu().numpy()

        bg_cpu = bg.cpu()
        bg_down = bg_cpu[:, :n_blocks * BLOCK].reshape(bg_cpu.shape[0], n_blocks, BLOCK).mean(dim=2)
        bg_np = bg_down[:1].numpy()
        explainer = shap.KernelExplainer(block_model_fn, bg_np, nsamples=10)
        samples_cpu = samples.cpu()
        samples_down = samples_cpu[:, :n_blocks * BLOCK].reshape(samples_cpu.shape[0], n_blocks, BLOCK).mean(dim=2)
        samples_np = samples_down.numpy()
        sv = explainer.shap_values(samples_np)
        sv = np.asarray(sv)
        if sv.ndim >= 2:
            sv = np.repeat(sv, BLOCK, axis=-1)
        if sv.ndim == 3:
            sv = sv.squeeze(-1)
        method = "KernelExplainer (downsampled)"
        print("  KernelExplainer succeeded.")

    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv.squeeze(-1)  # (n_samples, 64600)

    print(f"  Audio SHAP computed with {method}")
    return sv, samples, bg, float(explainer.expected_value[0])


def aggregate_embedding_importance(
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Rank embedding dimensions by mean |SHAP| across samples."""
    abs_mean = np.abs(shap_values).mean(axis=0)
    signed_mean = shap_values.mean(axis=0)
    df = pd.DataFrame({
        "dim": np.arange(len(abs_mean)),
        "mean_abs_shap": abs_mean,
        "mean_signed_shap": signed_mean,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def build_sample_explanations(
    shap_values: np.ndarray,
    embeddings: torch.Tensor,
    audio_files: List[Path],
    protocols: pd.DataFrame,
    model: nn.Module,
    offset: int = 0,
) -> pd.DataFrame:
    """Per-sample explanation table with prediction, ground truth,
    top positive/negative embedding dims, and confidence."""
    records = []
    for i in range(shap_values.shape[0]):
        file_idx = offset + i
        fname = audio_files[file_idx].stem

        row = protocols[protocols["filename"] == fname]
        gt = row["groundtruth"].values[0] if len(row) > 0 else "unknown"

        wav = audio_files[file_idx]
        waveform = load_audio(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = predict(model, waveform)
        prob = out["spoof_prob"].item()
        pred = "spoof" if prob >= 0.5 else "bonafide"
        correct = (pred == gt)

        sv = shap_values[i]
        top_pos = np.argsort(sv)[-5:][::-1]  # dims pushing toward spoof
        top_neg = np.argsort(sv)[:5]          # dims pushing toward bonafide

        records.append({
            "file": fname,
            "groundtruth": gt,
            "predicted": pred,
            "spoof_prob": round(prob, 4),
            "correct": correct,
            "top_spoof_dims": top_pos.tolist(),
            "top_bonafide_dims": top_neg.tolist(),
            "max_positive_shap": round(float(sv[top_pos[0]]), 6),
            "max_negative_shap": round(float(sv[top_neg[0]]), 6),
        })
    return pd.DataFrame(records)


# ===================================================================
# Visualization helpers
# ===================================================================

def plot_embedding_importance_bar(importance_df: pd.DataFrame, out_dir: Path, top_n: int = 20):
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top["mean_signed_shap"]]
    ax.barh(top["dim"].astype(str), top["mean_signed_shap"], color=colors)
    ax.set_xlabel("Mean SHAP value (red=spoof, blue=bonafide)")
    ax.set_ylabel("Embedding dimension")
    ax.set_title(f"Top {top_n} most influential embedding dimensions")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_dir / "embedding_importance_bar.png", dpi=150)
    plt.close(fig)


def plot_embedding_beeswarm(shap_values: np.ndarray, out_dir: Path, max_dims: int = 30):
    importance = np.abs(shap_values).mean(axis=0)
    top_dims = np.argsort(importance)[-max_dims:]
    data = shap_values[:, top_dims]
    fig, ax = plt.subplots(figsize=(12, 8))
    for j, d in enumerate(top_dims):
        jitter = np.random.normal(0, 0.15, size=data.shape[0])
        ax.scatter(data[:, j], np.full_like(data[:, j], j) + jitter,
                   s=6, alpha=0.5, c=data[:, j], cmap="RdBu_r", vmin=-np.max(np.abs(data)),
                   vmax=np.max(np.abs(data)))
    ax.set_yticks(range(len(top_dims)))
    ax.set_yticklabels([str(d) for d in top_dims])
    ax.set_xlabel("SHAP value")
    ax.set_title("Beeswarm: top embedding dimensions")
    ax.axvline(0, color="grey", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / "embedding_beeswarm.png", dpi=150)
    plt.close(fig)


def plot_audio_temporal_shap(shap_values: np.ndarray, out_dir: Path, sample_idx: int = 0):
    """Visualise which time-frames of the raw waveform drive the decision."""
    sv = shap_values[sample_idx]
    # Downsample for readability (mean-pool every 100 samples)
    ds = 100
    n_bins = len(sv) // ds
    sv_ds = sv[: n_bins * ds].reshape(n_bins, ds).mean(axis=1)
    time_axis = np.arange(n_bins) * ds / 16000  # assume 16 kHz

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(time_axis, sv_ds, width=ds / 16000 * 0.9,
           color=["#e74c3c" if v > 0 else "#3498db" for v in sv_ds])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SHAP value")
    ax.set_title(f"Temporal SHAP – which audio frames push toward spoof (red) or bonafide (blue)")
    ax.axhline(0, color="grey", linewidth=0.6)
    plt.tight_layout()
    fig.savefig(out_dir / f"temporal_shap_sample{sample_idx}.png", dpi=150)
    plt.close(fig)


def plot_audio_overview(shap_values: np.ndarray, out_dir: Path):
    """Overall summary of temporal SHAP magnitudes."""
    abs_sv = np.abs(shap_values)
    mean_abs = abs_sv.mean(axis=0)
    ds = 500
    n_bins = len(mean_abs) // ds
    mean_ds = mean_abs[: n_bins * ds].reshape(n_bins, ds).mean(axis=1)
    time_axis = np.arange(n_bins) * ds / 16000

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(time_axis, mean_ds, alpha=0.7, color="#8e44ad")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Average temporal importance across samples (which time segments matter most)")
    plt.tight_layout()
    fig.savefig(out_dir / "temporal_importance_overview.png", dpi=150)
    plt.close(fig)


def plot_correct_vs_incorrect(explanations: pd.DataFrame, importance_df: pd.DataFrame, out_dir: Path):
    """Compare aggregate feature importance for correct vs incorrect predictions."""
    if "correct" not in explanations.columns:
        return
    correct_files = set(explanations[explanations["correct"]]["file"])
    incorrect_files = set(explanations[~explanations["correct"]]["file"])

    summary = explanations.groupby("correct").agg(
        mean_spoof_prob=("spoof_prob", "mean"),
        count=("file", "count"),
    ).reset_index()
    summary["group"] = summary["correct"].map({True: "Correct", False: "Incorrect"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar: prediction confidence distribution
    for grp, color in [("Correct", "#2ecc71"), ("Incorrect", "#e74c3c")]:
        subset = explanations[explanations["correct"] == (grp == "Correct")]
        axes[0].hist(subset["spoof_prob"], bins=20, alpha=0.6, label=grp, color=color)
    axes[0].set_xlabel("Spoof probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction confidence distribution")
    axes[0].legend()

    # Table summary
    axes[1].axis("off")
    table_data = summary[["group", "mean_spoof_prob", "count"]].values.tolist()
    table = axes[1].table(
        cellText=table_data,
        colLabels=["Group", "Mean Spoof Prob", "N samples"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    axes[1].set_title("Prediction summary")

    plt.tight_layout()
    fig.savefig(out_dir / "correct_vs_incorrect.png", dpi=150)
    plt.close(fig)


def plot_confidence_histogram(explanations: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for gt_label, color in [("spoof", "#e74c3c"), ("bonafide", "#3498db")]:
        subset = explanations[explanations["groundtruth"] == gt_label]
        ax.hist(subset["spoof_prob"], bins=25, alpha=0.6, label=gt_label, color=color)
    ax.set_xlabel("Model spoof probability")
    ax.set_ylabel("Count")
    ax.set_title("Spoof probability distribution by true class")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "confidence_histogram.png", dpi=150)
    plt.close(fig)


def plot_occlusion_per_sample(importance: np.ndarray, n_windows: int,
                              out_dir: Path, n_show: int = 3):
    """Bar chart of occlusion importance for the first few samples."""
    sr = 16000
    window_len = TARGET_LENGTH // n_windows
    time_axis = np.arange(n_windows) * window_len / sr

    for i in range(min(importance.shape[0], n_show)):
        vals = importance[i]
        fig, ax = plt.subplots(figsize=(14, 4))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
        ax.bar(time_axis, vals, width=window_len / sr * 0.9, color=colors)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Delta P(spoof)")
        ax.set_title(f"Occlusion sensitivity – sample {i}  "
                     f"(red=removing hurts spoof, blue=removing hurts bonafide)")
        ax.axhline(0, color="grey", linewidth=0.6)
        plt.tight_layout()
        fig.savefig(out_dir / f"occlusion_sample{i}.png", dpi=150)
        plt.close(fig)


def plot_occlusion_aggregate(importance: np.ndarray, n_windows: int,
                             out_dir: Path):
    """Mean |delta| across all samples – which time segments matter most."""
    sr = 16000
    window_len = TARGET_LENGTH // n_windows
    time_axis = np.arange(n_windows) * window_len / sr
    mean_abs = np.abs(importance).mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(time_axis, mean_abs, alpha=0.7, color="#8e44ad")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean |Delta P(spoof)|")
    ax.set_title("Average occlusion importance across samples "
                 "(which time segments matter most)")
    plt.tight_layout()
    fig.savefig(out_dir / "occlusion_aggregate.png", dpi=150)
    plt.close(fig)


def plot_occlusion_spoof_vs_bonafide(importance: np.ndarray,
                                     protocols: pd.DataFrame,
                                     audio_files: List[Path],
                                     n_windows: int,
                                     out_dir: Path):
    """Compare mean occlusion importance for spoof vs bonafide files."""
    sr = 16000
    window_len = TARGET_LENGTH // n_windows
    time_axis = np.arange(n_windows) * window_len / sr

    spoof_idx, bonafide_idx = [], []
    for i, af in enumerate(audio_files):
        fname = af.stem
        row = protocols[protocols["filename"] == fname]
        gt = row["groundtruth"].values[0] if len(row) > 0 else "unknown"
        if gt == "spoof":
            spoof_idx.append(i)
        else:
            bonafide_idx.append(i)

    fig, ax = plt.subplots(figsize=(14, 4))
    if spoof_idx:
        mean_spoof = np.abs(importance[spoof_idx]).mean(axis=0)
        ax.plot(time_axis, mean_spoof, color="#e74c3c", label="Spoof", linewidth=1.5)
    if bonafide_idx:
        mean_bona = np.abs(importance[bonafide_idx]).mean(axis=0)
        ax.plot(time_axis, mean_bona, color="#3498db", label="Bonafide", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean |Delta P(spoof)|")
    ax.set_title("Occlusion importance: spoof vs bonafide files")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "occlusion_spoof_vs_bonafide.png", dpi=150)
    plt.close(fig)


def plot_spectrogram_shap_per_sample(
    shap_values: np.ndarray,
    out_dir: Path,
    n_mels: int,
    n_frames: int,
    n_show: int = 3,
    sr: int = SR,
    hop_length: int = MEL_HOP_LENGTH,
    n_fft: int = MEL_N_FFT,
):
    """Heatmap of SHAP values shaped as (mel freq x time) for a few samples."""
    for i in range(min(shap_values.shape[0], n_show)):
        sv = shap_values[i]  # (n_mels, n_frames)
        duration = n_frames * hop_length / sr
        time_axis = np.linspace(0, duration, n_frames)
        freq_axis = np.arange(n_mels)

        fig, ax = plt.subplots(figsize=(14, 5))
        vmax = np.max(np.abs(sv))
        im = ax.imshow(sv, aspect="auto", origin="lower",
                       extent=[0, duration, 0, n_mels],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel frequency bin")
        ax.set_title(f"Spectrogram SHAP – sample {i}  "
                     f"(red=spoof, blue=bonafide)")
        plt.colorbar(im, ax=ax, label="SHAP value")
        plt.tight_layout()
        fig.savefig(out_dir / f"spectrogram_shap_sample{i}.png", dpi=150)
        plt.close(fig)


def plot_spectrogram_shap_aggregate(
    shap_values: np.ndarray,
    out_dir: Path,
    n_mels: int,
    n_frames: int,
    sr: int = SR,
    hop_length: int = MEL_HOP_LENGTH,
):
    """Mean |SHAP| across samples, shown as a spectrogram-shaped heatmap."""
    mean_abs = np.abs(shap_values).mean(axis=0)  # (n_mels, n_frames)
    duration = n_frames * hop_length / sr

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(mean_abs, aspect="auto", origin="lower",
                   extent=[0, duration, 0, n_mels],
                   cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel frequency bin")
    ax.set_title("Mean |SHAP| across samples – which time-frequency regions matter most")
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    plt.tight_layout()
    fig.savefig(out_dir / "spectrogram_shap_aggregate.png", dpi=150)
    plt.close(fig)


def plot_spectrogram_shap_top_regions(
    shap_values: np.ndarray,
    out_dir: Path,
    n_mels: int,
    n_frames: int,
    sr: int = SR,
    hop_length: int = MEL_HOP_LENGTH,
):
    """Top time-frequency regions by mean |SHAP|, shown as a ranked bar chart."""
    mean_abs = np.abs(shap_values).mean(axis=0)  # (n_mels, n_frames)
    duration = n_frames * hop_length / sr

    # Flatten and find top 20 regions
    flat = mean_abs.flatten()
    top_idx = np.argsort(flat)[-20:][::-1]
    top_mel = top_idx // n_frames
    top_frame = top_idx % n_frames
    top_vals = flat[top_idx]
    top_times = top_frame * hop_length / sr

    labels = [f"mel={m}, {t:.2f}s" for m, t in zip(top_mel, top_times)]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.magma(np.linspace(0.3, 0.9, len(top_vals)))
    ax.barh(range(len(top_vals)), top_vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Top 20 time-frequency regions by importance")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_dir / "spectrogram_shap_top_regions.png", dpi=150)
    plt.close(fig)


def plot_spectrogram_shap_spoof_vs_bonafide(
    shap_values: np.ndarray,
    protocols: pd.DataFrame,
    audio_files: List[Path],
    out_dir: Path,
    n_mels: int,
    n_frames: int,
    sr: int = SR,
    hop_length: int = MEL_HOP_LENGTH,
):
    """Compare mean |SHAP| for spoof vs bonafide, summed over frequency."""
    duration = n_frames * hop_length / sr
    time_axis = np.linspace(0, duration, n_frames)

    spoof_idx, bonafide_idx = [], []
    for i, af in enumerate(audio_files):
        fname = af.stem
        row = protocols[protocols["filename"] == fname]
        gt = row["groundtruth"].values[0] if len(row) > 0 else "unknown"
        if gt == "spoof":
            spoof_idx.append(i)
        else:
            bonafide_idx.append(i)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Time-collapsed importance
    if spoof_idx:
        mean_spoof = np.abs(shap_values[spoof_idx]).mean(axis=(0, 1))
        axes[0].plot(time_axis, mean_spoof, color="#e74c3c", label="Spoof", linewidth=1.5)
    if bonafide_idx:
        mean_bona = np.abs(shap_values[bonafide_idx]).mean(axis=(0, 1))
        axes[0].plot(time_axis, mean_bona, color="#3498db", label="Bonafide", linewidth=1.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Mean |SHAP| (collapsed over freq)")
    axes[0].set_title("Time-collapsed importance")
    axes[0].legend()

    # Frequency-collapsed importance
    freq_axis = np.arange(n_mels)
    if spoof_idx:
        mean_spoof_f = np.abs(shap_values[spoof_idx]).mean(axis=(0, 2))
        axes[1].plot(freq_axis, mean_spoof_f, color="#e74c3c", label="Spoof", linewidth=1.5)
    if bonafide_idx:
        mean_bona_f = np.abs(shap_values[bonafide_idx]).mean(axis=(0, 2))
        axes[1].plot(freq_axis, mean_bona_f, color="#3498db", label="Bonafide", linewidth=1.5)
    axes[1].set_xlabel("Mel frequency bin")
    axes[1].set_ylabel("Mean |SHAP| (collapsed over time)")
    axes[1].set_title("Frequency-collapsed importance")
    axes[1].legend()

    plt.suptitle("Spectrogram SHAP: spoof vs bonafide", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "spectrogram_shap_spoof_vs_bonafide.png", dpi=150)
    plt.close(fig)


# ===================================================================
# Natural-language summary
# ===================================================================

def print_interpretation_summary(
    importance_df: pd.DataFrame,
    shap_values: np.ndarray,
    explanations: pd.DataFrame,
    expected_value: float,
    occl_importance: np.ndarray = None,
    protocols: pd.DataFrame = None,
    audio_files: List[Path] = None,
):
    print("\n" + "=" * 70)
    print("  AASIST INTERPRETATION SUMMARY")
    print("=" * 70)

    n_correct = explanations["correct"].sum()
    n_total = len(explanations)
    print(f"\n  Accuracy on explained samples: {n_correct}/{n_total} "
          f"({100 * n_correct / n_total:.1f}%)")

    avg_prob = explanations["spoof_prob"].mean()
    print(f"  Mean spoof probability: {avg_prob:.4f}")
    print(f"  SHAP expected value (logit baseline): {expected_value:.4f}")

    print("\n  Top-5 embedding dimensions driving SPOOF predictions:")
    for _, row in importance_df.head(5).iterrows():
        print(f"    dim {int(row['dim']):>3d}  |  mean |SHAP| = {row['mean_abs_shap']:.6f}"
              f"  |  signed = {row['mean_signed_shap']:+.6f}")

    print("\n  Top-5 embedding dimensions driving BONAFIDE predictions:")
    tail = importance_df.tail(5).iloc[::-1]
    for _, row in tail.iterrows():
        print(f"    dim {int(row['dim']):>3d}  |  mean |SHAP| = {row['mean_abs_shap']:.6f}"
              f"  |  signed = {row['mean_signed_shap']:+.6f}")

    spoof_dims = np.where(shap_values.mean(axis=0) > 0)[0]
    bonafide_dims = np.where(shap_values.mean(axis=0) < 0)[0]
    print(f"\n  Dimensions consistently pushing toward spoof: {len(spoof_dims)}")
    print(f"  Dimensions consistently pushing toward bonafide: {len(bonafide_dims)}")

    if len(explanations[~explanations["correct"]]) > 0:
        print("\n  Misclassified samples (potential model weaknesses):")
        for _, row in explanations[~explanations["correct"]].head(5).iterrows():
            print(f"    {row['file']}  |  GT: {row['groundtruth']}  |  "
                  f"Pred: {row['predicted']}  |  P(spoof)={row['spoof_prob']:.4f}")

    if occl_importance is not None and protocols is not None and audio_files is not None:
        sr = 16000
        window_len = TARGET_LENGTH // occl_importance.shape[1]
        mean_abs = np.abs(occl_importance).mean(axis=0)
        top_windows = np.argsort(mean_abs)[-5:][::-1]
        print("\n  Top-5 time windows by occlusion importance (mean |Delta P(spoof)|):")
        for w in top_windows:
            t_start = w * window_len / sr
            t_end = (w + 1) * window_len / sr
            print(f"    window {w:>3d}  |  {t_start:.2f}s - {t_end:.2f}s  |  "
                  f"mean |Delta| = {mean_abs[w]:.6f}")

    print("\n" + "=" * 70)


# ===================================================================
# Main pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Interpret AASIST v3 with SHAP")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .flac audio files", default=r"C:\Users\eglha\.cache\kagglehub\datasets\mohammedabdeldayem\avsspoof-2021\versions\7\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac")
    parser.add_argument("--protocol", type=str, required=True,
                        help="Path to ASVspoof2019 protocol .txt file", default=r"C:\Users\eglha\.cache\kagglehub\datasets\mohammedabdeldayem\avsspoof-2021\versions\7\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\ASVspoof2021.DF.cm.eval.trl.txt")
    parser.add_argument("--n_background", type=int, default=100,
                        help="Number of background samples for SHAP")
    parser.add_argument("--n_embed_samples", type=int, default=50,
                        help="Number of samples for embedding-level SHAP")
    parser.add_argument("--n_audio_samples", type=int, default=10,
                        help="Number of samples for temporal SHAP (keep small)")
    parser.add_argument("--n_occl_windows", type=int, default=100,
                        help="Number of windows for occlusion sensitivity")
    parser.add_argument("--out_dir", type=str,
                        default="interpretation_results",
                        help="Directory for output plots and CSVs")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Override device (cuda/cpu)")
    args = parser.parse_args()

    if args.device:
        global DEVICE
        DEVICE = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading AASIST v3 on {DEVICE}...")
    model = load_aasist_v3(device=str(DEVICE))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({n_params:,} params)")

    # ------------------------------------------------------------------
    # 2. Load audio + protocol
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    audio_files = sorted(data_dir.glob("*.flac"))
    random.seed(42)
    random.shuffle(audio_files)
    max_needed = args.n_background + args.n_embed_samples + args.n_audio_samples
    audio_files = audio_files[:max_needed]
    print(f"Using {len(audio_files)} audio files from {data_dir}")

    protocols = parse_protocol(args.protocol)
    print(f"Protocol loaded ({len(protocols)} entries)")

    # ------------------------------------------------------------------
    # 3. Extract embeddings + audio tensors
    # ------------------------------------------------------------------
    embeddings = []
    audio_tensors = []
    for af in tqdm(audio_files, desc="Extracting embeddings"):
        wav = load_audio(af).to(DEVICE)
        with torch.no_grad():
            out = predict(model, wav.unsqueeze(0))
        embeddings.append(out["embedding"].squeeze(0).cpu())
        audio_tensors.append(wav.cpu())

    embeddings = torch.stack(embeddings)
    audio_tensors = torch.stack(audio_tensors)
    print(f"Embeddings shape: {embeddings.shape}")

    # ------------------------------------------------------------------
    # 4. Embedding-level SHAP
    # ------------------------------------------------------------------
    print("\n--- Embedding-level SHAP ---")
    embed_shap, embed_samples, embed_bg, embed_ev = compute_embedding_shap(
        model, embeddings,
        background_n=min(args.n_background, len(embeddings)),
        n_samples=min(args.n_embed_samples, len(embeddings) - args.n_background),
    )
    print(f"Embedding SHAP shape: {embed_shap.shape}")

    importance_df = aggregate_embedding_importance(embed_shap)
    importance_df.to_csv(out_dir / "embedding_importance.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Temporal occlusion sensitivity (replaces SHAP-based temporal)
    # ------------------------------------------------------------------
    print("\n--- Temporal occlusion sensitivity ---")
    n_occl_windows = args.n_occl_windows
    occl_importance, occl_baseline = compute_occlusion_sensitivity(
        model, audio_tensors, n_windows=n_occl_windows,
    )
    print(f"Occlusion importance shape: {occl_importance.shape}")

    occl_df = pd.DataFrame(occl_importance)
    occl_df.columns = [f"w{i}" for i in range(n_occl_windows)]
    occl_df.insert(0, "file", [af.stem for af in audio_files])
    occl_df.insert(1, "baseline_spoof_prob", occl_baseline)
    occl_df.to_csv(out_dir / "occlusion_sensitivity.csv", index=False)

    # ------------------------------------------------------------------
    # 5b. Spectrogram surrogate SHAP
    # ------------------------------------------------------------------
    print("\n--- Spectrogram surrogate SHAP ---")
    spec_shap, all_specs, spec_ev, n_mels_out, n_frames = compute_spectrogram_shap(
        model, audio_tensors,
        background_n=min(args.n_background, len(audio_tensors)),
        n_samples=min(args.n_embed_samples, len(audio_tensors) - args.n_background),
    )
    print(f"Spectrogram SHAP shape: {spec_shap.shape}")

    np.save(out_dir / "spectrogram_shap_values.npy", spec_shap)
    np.save(out_dir / "spectrogram_values.npy", all_specs)

    # ------------------------------------------------------------------
    # 6. Build per-sample explanation table
    # ------------------------------------------------------------------
    explanations = build_sample_explanations(
        embed_shap, embeddings, audio_files, protocols, model,
        offset=args.n_background,
    )
    explanations.to_csv(out_dir / "sample_explanations.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_embedding_importance_bar(importance_df, out_dir)
    plot_embedding_beeswarm(embed_shap, out_dir)
    plot_confidence_histogram(explanations, out_dir)
    plot_correct_vs_incorrect(explanations, importance_df, out_dir)
    plot_occlusion_per_sample(occl_importance, n_occl_windows, out_dir)
    plot_occlusion_aggregate(occl_importance, n_occl_windows, out_dir)
    plot_occlusion_spoof_vs_bonafide(
        occl_importance, protocols, audio_files, n_occl_windows, out_dir,
    )
    plot_spectrogram_shap_per_sample(spec_shap, out_dir, n_mels_out, n_frames)
    plot_spectrogram_shap_aggregate(spec_shap, out_dir, n_mels_out, n_frames)
    plot_spectrogram_shap_top_regions(spec_shap, out_dir, n_mels_out, n_frames)
    plot_spectrogram_shap_spoof_vs_bonafide(
        spec_shap, protocols,
        audio_files[args.n_background:args.n_background + args.n_embed_samples],
        out_dir, n_mels_out, n_frames,
    )

    # ------------------------------------------------------------------
    # 8. Print human-readable summary
    # ------------------------------------------------------------------
    print_interpretation_summary(
        importance_df, embed_shap, explanations, embed_ev,
        occl_importance=occl_importance, protocols=protocols, audio_files=audio_files,
    )

    print(f"\nAll results saved to: {out_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
