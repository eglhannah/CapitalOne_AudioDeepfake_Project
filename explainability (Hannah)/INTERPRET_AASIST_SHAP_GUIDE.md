# AASIST SHAP Interpretation Guide

`interpret_aasist_shap.py` explains **why** the AASIST v3 model classifies audio as spoof or bonafide by applying SHAP (SHapley Additive exPlanations) at three complementary levels of analysis.

---

## Overview

AASIST is a graph attention network that takes a raw audio waveform (64,600 samples at 16 kHz) and produces a 128-dimensional embedding, which is then passed through a linear classification head to yield a spoof/bonafide prediction. This script peels back each layer of that pipeline using SHAP to answer:

- **Which embedding dimensions** are most responsible for the model's decision?
- **Which time-frames** in the original audio waveform matter most?
- **How do explanations differ** between correct and incorrect predictions?

---

## Three Levels of Analysis

### 1. Embedding-Level SHAP

**What it does:** Wraps only the final classification head (`out_layer`) and computes `DeepExplainer` SHAP values over the 128-d embedding vector. This tells you which abstract features learned by the graph attention layers push the prediction toward spoof vs bonafide.

**Why it matters:** The embedding is the model's compressed "understanding" of the audio. Identifying the most influential dimensions helps you understand what the model has learned to look for.

**Outputs:**
| File | Description |
|---|---|
| `embedding_importance.csv` | All 128 dimensions ranked by mean \|SHAP\|, with signed mean |
| `embedding_importance_bar.png` | Top-20 dimensions by signed SHAP (red = spoof, blue = bonafide) |
| `embedding_beeswarm.png` | Scatter plot showing per-sample SHAP values for top dimensions |

### 2. Temporal (Raw Audio) SHAP

**What it does:** Wraps the **entire** AASIST model and computes SHAP values over the raw 64,600-sample waveform. This reveals which segments of the audio signal are most important for the classification decision.

**Why it matters:** This is the most interpretable level — you can see whether the model is focusing on the beginning, middle, or end of the audio, or on specific transient events. Spoofed audio often has artifacts in specific time regions.

**Note:** This is computationally expensive (64,600 features). Default sample counts are intentionally small (5-10 samples).

**Outputs:**
| File | Description |
|---|---|
| `temporal_shap_sample*.png` | Per-sample time-series of SHAP values (red = spoof evidence, blue = bonafide evidence) |
| `temporal_importance_overview.png` | Aggregate \|SHAP\| envelope across all temporal samples — shows which time regions consistently matter |

### 3. Aggregate & Correctness Analysis

**What it does:** Combines predictions across all explained samples to compare correct vs incorrect classifications, examine the model's confidence distribution, and identify systematic weaknesses.

**Outputs:**
| File | Description |
|---|---|
| `sample_explanations.csv` | Per-file table: ground truth, prediction, P(spoof), top spoof/bonafide dims, correctness |
| `confidence_histogram.png` | Distribution of spoof probability by true class (spoof vs bonafide) |
| `correct_vs_incorrect.png` | Side-by-side: confidence distribution + summary table for correct vs misclassified |

---

## Interpreting the Output

### Embedding Importance Plot (`embedding_importance_bar.png`)

- **Red bars (positive SHAP):** Dimensions that push the prediction toward **spoof**
- **Blue bars (negative SHAP):** Dimensions that push the prediction toward **bonafide**
- A dimension with large absolute SHAP but near-zero signed mean is important but inconsistent across samples

### Temporal SHAP Plot (`temporal_shap_sample*.png`)

- **Red regions:** Time frames providing evidence of **spoof**
- **Blue regions:** Time frames providing evidence of **bonafide**
- Large magnitude in a narrow band suggests a specific artifact (e.g., a codec signature or splicing boundary)

### Summary Console Output

The script prints a structured summary including:
- Accuracy on explained samples
- Top-5 embedding dimensions driving spoof predictions
- Top-5 embedding dimensions driving bonafide predictions
- Count of dimensions consistently favoring each class
- List of misclassified samples (model weaknesses)

---

## Usage

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
| `--n_audio_samples` | `10` | Samples for temporal SHAP (keep low; 64k features) |
| `--out_dir` | `interpretation_results` | Output directory for plots and CSVs |
| `--device` | `cuda`/`cpu` auto | Override compute device |

---

## Key Dependencies

- `shap` — SHAP library (DeepExplainer)
- `torch` — PyTorch (model inference + gradient tracing)
- `librosa` — audio feature extraction (in companion analysis)
- `matplotlib`, `seaborn` — plotting
- `soundfile` — FLAC/WAV reading
- `aasist.simple_aasist` — local AASIST v3 loader from this repo

---

## Relationship to Other Files

| File | Role |
|---|---|
| `init_aasistSHAP.py` | Initializes model, loads predictions, prepares background/samples for interactive use |
| `aasist_shap_embeddings.py` | Earlier embedding-level SHAP exploration (notebook-style) |
| `aasist_shap.py` | Audio feature extraction + PCA visualization |
| **`interpret_aasist_shap.py`** | **This file** — comprehensive, scriptable interpretation pipeline |

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
├── temporal_shap_sample0.png
├── temporal_shap_sample1.png
├── temporal_shap_sample2.png
└── temporal_importance_overview.png
```
