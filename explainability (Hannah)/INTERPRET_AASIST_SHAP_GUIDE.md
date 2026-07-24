# AASIST SHAP Interpretation Guide

`interpret_aasist_shap.py` explains **why** the AASIST v3 model classifies audio as spoof or bonafide by applying SHAP (SHapley Additive exPlanations) at three complementary levels of analysis. `dash_aasist_app.py` provides an interactive Dash dashboard for single-file prediction and explanation.

---

## Overview

AASIST is a graph attention network that takes a raw audio waveform (64,600 samples at 16 kHz) and produces a 128-dimensional embedding, which is then passed through a linear classification head to yield a spoof/bonafide prediction. This script peels back each layer of that pipeline using SHAP to answer:

- **Which embedding dimensions** are most responsible for the model's decision?
- **Which time-windows** in the original audio waveform matter most?
- **Which spectrogram regions** contribute most to the spoof/bonafide decision?

---

## Three Levels of Analysis

### 1. Embedding-Level SHAP

**What it does:** Wraps only the final classification head (`out_layer`) and computes `DeepExplainer` SHAP values over the 128-d embedding vector. This tells you which abstract features learned by the graph attention layers push the prediction toward spoof vs bonafide.

**Background strategy:** Since SHAP requires a background distribution to compute expected values, the embedding is replicated 20 times and small Gaussian noise (5% of embedding magnitude) is added to create a perturbation-based background set. This ensures SHAP values are non-trivial (a single-sample background would yield all-zero SHAP values because the expected value equals the input).

**Why it matters:** The embedding is the model's compressed "understanding" of the audio. Identifying the most influential dimensions helps you understand what the model has learned to look for.

**Outputs:**
| File | Description |
|---|---|
| `embedding_importance.csv` | All 128 dimensions ranked by mean \|SHAP\|, with signed mean |
| `embedding_importance_bar.png` | Top-20 dimensions by signed SHAP (red = spoof, blue = bonafide) |
| `embedding_beeswarm.png` | Scatter plot showing per-sample SHAP values for top dimensions |

### 2. Occlusion Sensitivity (Temporal Analysis)

**What it does:** Divides the 64,600-sample waveform into 100 equal windows. Each window is zeroed out in turn and the change in spoof probability is recorded. A large positive delta means that window was important for the spoof decision; a large negative delta means it was important for the bonafide decision.

**Why it replaces raw-audio SHAP:** SHAP over the full 64,600-sample waveform is computationally prohibitive (64,600 features) and often fails due to in-place operations (`SELU(inplace=True)`, `+=` in residual blocks) that break SHAP's gradient tracing. Occlusion sensitivity is model-agnostic, requires no gradient computation, and produces interpretable time-aligned importance scores.

**Outputs:**
| File | Description |
|---|---|
| `occlusion_per_sample.png` | Per-sample bar chart of importance by time window |
| `occlusion_aggregate.png` | Mean importance across all samples |
| `occlusion_spoof_vs_bonafide.png` | Side-by-side: mean importance for spoof vs bonafide samples |

### 3. Spectrogram SHAP (Surrogate MLP)

**What it does:** Trains a small MLP surrogate (`SpectrogramSurrogate`: 5080 → 256 → 64 → 1) to predict the AASIST spoof probability from a flattened 40-band log-mel spectrogram. Then runs `DeepExplainer` SHAP on the surrogate to produce a time-frequency attribution heatmap.

**Why a surrogate?** AASIST's in-place operations prevent direct SHAP on the full model. The surrogate bypasses this by learning a simple function from spectrograms to spoof probability, then explaining that function. The surrogate typically achieves >0.98 correlation with AASIST predictions.

**Background strategy:** 20 perturbed copies of the spectrogram (5% Gaussian noise) serve as the SHAP background, same as embedding SHAP.

**Parameters:**
- 40 mel frequency bins, 1024 FFT, 512 hop length → shape (40, 127) = 5,080 features
- Surrogate: 200 training epochs, Adam optimizer, MSE loss

**Outputs:**
| File | Description |
|---|---|
| `spectrogram_shap_sample*.png` | Per-sample time-frequency SHAP heatmap (red = spoof, blue = bonafide) |
| `spectrogram_shap_aggregate.png` | Mean SHAP heatmap across all samples |
| `spectrogram_shap_top_regions.png` | Bar chart of top regions by aggregate \|SHAP\| |
| `spectrogram_shap_spoof_vs_bonafide.png` | Side-by-side heatmaps for spoof vs bonafide samples |

---

## Interpreting the Output

### Embedding Importance Plot (`embedding_importance_bar.png`)

- **Red bars (positive SHAP):** Dimensions that push the prediction toward **spoof**
- **Blue bars (negative SHAP):** Dimensions that push the prediction toward **bonafide**
- A dimension with large absolute SHAP but near-zero signed mean is important but inconsistent across samples

### Occlusion Sensitivity Plot (`occlusion_per_sample.png`)

- **Red bars:** Windows where zeroing increased the spoof decision (important for bonafide evidence)
- **Blue bars:** Windows where zeroing decreased the spoof decision (important for spoof evidence)
- Large magnitude in a narrow band suggests a specific artifact (e.g., codec signature or splicing boundary)

### Spectrogram SHAP Heatmap (`spectrogram_shap_sample*.png`)

- **Red regions:** Time-frequency bins providing evidence of **spoof**
- **Blue regions:** Time-frequency bins providing evidence of **bonafide**
- Concentrated hot-spots may correspond to codec artifacts, phase discontinuities, or frequency-specific anomalies

### Summary Console Output

The script prints a structured summary including:
- Surrogate-AASIST correlation
- Accuracy on explained samples
- Top-5 embedding dimensions driving spoof predictions
- Top-5 embedding dimensions driving bonafide predictions
- Count of dimensions consistently favoring each class
- List of misclassified samples (model weaknesses)

---

## Dash Dashboard (`dash_aasist_app.py`)

An interactive web dashboard for single-file prediction and explanation, built with Dash and Plotly.

### Running

```bash
python "explainability (Hannah)/dash_aasist_app.py"
```

Then open http://127.0.0.1:8050 in your browser.

### Features

- **Audio upload**: Supports `.flac`, `.wav`, `.mp3`, `.aac`, `.m4a`, `.ogg`, `.wma`, `.opus`
- **Audio playback**: Built-in player for uploaded audio
- **Prediction card**: Spoof/bonafide classification with confidence
- **Mel spectrogram**: Log-mel spectrogram visualization (40 mel bins)
- **Embedding SHAP**: Top-20 dimensions bar chart (same as batch script)
- **Occlusion sensitivity**: Per-window importance timeline (same as batch script)
- **Spectrogram SHAP**: Time-frequency attribution heatmap (same as batch script)
- **Status indicators**: "Loading model..." → "Ready for audio file" → "Analyzing..." → "Analysis complete"
- **Error handling**: Failed analyses show error messages on the chart instead of blank panels
- **Darkly theme**: Dark-themed UI via Bootstrap

### Audio Loading

The dashboard uses `imageio-ffmpeg` (which bundles its own ffmpeg binary) to decode compressed audio formats via subprocess, bypassing pydub. This avoids the conda ffmpeg DLL dependency issues on Windows. For FLAC and WAV files, `soundfile` is used directly.

### Dependencies

```bash
pip install dash dash-bootstrap-components plotly imageio-ffmpeg
```

---

## Batch Script Usage (`interpret_aasist_shap.py`)

```bash
python "explainability (Hannah)/interpret_aasist_shap.py" \
  --data_dir "I:\My Drive\ASVSpoof_Data\unzipped2019\LA\LA\ASVspoof2019_LA_dev\flac" \
  --protocol "I:\My Drive\ASVSpoof_Data\unzipped2019\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Directory containing `.flac` audio files |
| `--protocol` | *(required)* | Path to ASVspoof2019 protocol `.txt` file |
| `--n_background` | `100` | Background samples for SHAP baseline (embedding level) |
| `--n_embed_samples` | `50` | Samples to explain at the embedding level |
| `--n_occlusion_windows` | `100` | Number of time windows for occlusion sensitivity |
| `--n_spec_samples` | `50` | Samples for spectrogram SHAP |
| `--out_dir` | `interpretation_results` | Output directory for plots and CSVs |
| `--device` | `cuda`/`cpu` auto | Override compute device |

---

## Key Dependencies

- `shap` — SHAP library (DeepExplainer)
- `torch` — PyTorch (model inference + gradient tracing)
- `torchaudio` — mel spectrogram computation
- `matplotlib`, `seaborn` — plotting
- `soundfile` — FLAC/WAV reading
- `imageio-ffmpeg` — bundled ffmpeg for compressed audio (m4a, mp3, etc.)
- `dash`, `dash-bootstrap-components`, `plotly` — interactive dashboard
- `aasist.simple_aasist` — local AASIST v3 loader from this repo

---

## Relationship to Other Files

| File | Role |
|---|---|
| **`interpret_aasist_shap.py`** | **Core library** — all explanation functions, batch script, and plotting |
| **`dash_aasist_app.py`** | **Interactive dashboard** — single-file prediction + explanation via web UI |
| `init_aasistSHAP.py` | Initializes model, loads predictions, prepares background/samples for interactive use |
| `aasist_shap_embeddings.py` | Earlier embedding-level SHAP exploration (notebook-style) |
| `aasist_shap.py` | Audio feature extraction + PCA visualization |

---

## Output Directory Structure

```
interpretation_results/
├── embedding_importance.csv
├── embedding_importance_bar.png
├── embedding_beeswarm.png
├── sample_explanations.csv
├── confidence_histogram.png
├── correct_vs_incorrect.png
├── occlusion_per_sample.png
├── occlusion_aggregate.png
├── occlusion_spoof_vs_bonafide.png
├── spectrogram_shap_sample0.png
├── spectrogram_shap_sample1.png
├── spectrogram_shap_sample2.png
├── spectrogram_shap_aggregate.png
├── spectrogram_shap_top_regions.png
└── spectrogram_shap_spoof_vs_bonafide.png
```

---

## Design Decisions

### Why Occlusion Sensitivity Instead of Raw-Audio SHAP?

SHAP's `DeepExplainer`, `GradientExplainer`, and `KernelExplainer` all fail on the full AASIST model due to:
1. **In-place operations** — `SELU(inplace=True)` and `+=` in residual blocks corrupt the computation graph
2. **Memory** — 64,600 features × batch size exceeds GPU memory
3. **Speed** — Even when it works, raw-audio SHAP takes minutes per sample

Occlusion sensitivity is model-agnostic, produces interpretable time-aligned scores, and runs in seconds per sample.

### Why a Surrogate for Spectrogram SHAP?

Direct SHAP on AASIST fails (same in-place op issues). The surrogate MLP provides a differentiable, SHAP-compatible function that approximates AASIST's decision boundary in spectrogram space. With >0.98 correlation, the surrogate's explanations faithfully represent the model's reasoning.

### Why Perturbation-Based Backgrounds?

SHAP's `DeepExplainer` computes values as deviations from the expected output under a background distribution. With a single background sample, the expected value equals the input value, producing all-zero SHAP values. Adding small Gaussian perturbations (5% of magnitude) creates a realistic background distribution while staying close to the original data manifold.
