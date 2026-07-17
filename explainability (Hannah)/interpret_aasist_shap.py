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
import seaborn as sns
import soundfile as sf

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
    128-d embedding that the GAT pipeline produces."""

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

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        _, logits = self.model(audio)
        return logits[:, 1].unsqueeze(1)


# ===================================================================
# Core analysis functions
# ===================================================================

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

    return sv, samples, bg, float(explainer.expected_value[0])


def compute_audio_shap(
    model: nn.Module,
    audio_tensors: torch.Tensor,
    background_n: int = 30,
    n_samples: int = 10,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, float]:
    """SHAP DeepExplainer over raw waveform samples.

    Because the audio tensor is 64 600 dims this is *expensive*; we keep
    sample counts deliberately small.
    """
    wrapper = AASISTAudioWrapper(model).to(DEVICE).eval()
    bg = audio_tensors[:background_n].to(DEVICE)
    samples = audio_tensors[background_n : background_n + n_samples].to(DEVICE)

    explainer = shap.DeepExplainer(wrapper, bg)
    sv = explainer.shap_values(samples, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv.squeeze(-1)  # (n_samples, 64600)

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


# ===================================================================
# Natural-language summary
# ===================================================================

def print_interpretation_summary(
    importance_df: pd.DataFrame,
    shap_values: np.ndarray,
    explanations: pd.DataFrame,
    expected_value: float,
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

    print("\n" + "=" * 70)


# ===================================================================
# Main pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Interpret AASIST v3 with SHAP")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .flac audio files")
    parser.add_argument("--protocol", type=str, required=True,
                        help="Path to ASVspoof2019 protocol .txt file")
    parser.add_argument("--n_background", type=int, default=100,
                        help="Number of background samples for SHAP")
    parser.add_argument("--n_embed_samples", type=int, default=50,
                        help="Number of samples for embedding-level SHAP")
    parser.add_argument("--n_audio_samples", type=int, default=10,
                        help="Number of samples for temporal SHAP (keep small)")
    parser.add_argument("--out_dir", type=str,
                        default="interpretation_results",
                        help="Directory for output plots and CSVs")
    parser.add_argument("--device", type=str, default=None,
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
    # 5. Temporal (raw audio) SHAP
    # ------------------------------------------------------------------
    print("\n--- Temporal SHAP (raw waveform) ---")
    audio_shap, audio_samples, audio_bg, audio_ev = compute_audio_shap(
        model, audio_tensors,
        background_n=min(args.n_audio_samples, len(audio_tensors)),
        n_samples=min(5, len(audio_tensors) - args.n_audio_samples),
    )
    print(f"Temporal SHAP shape: {audio_shap.shape}")

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

    for i in range(min(audio_shap.shape[0], 3)):
        plot_audio_temporal_shap(audio_shap, out_dir, sample_idx=i)
    plot_audio_overview(audio_shap, out_dir)

    # ------------------------------------------------------------------
    # 8. Print human-readable summary
    # ------------------------------------------------------------------
    print_interpretation_summary(importance_df, embed_shap, explanations, embed_ev)

    print(f"\nAll results saved to: {out_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
