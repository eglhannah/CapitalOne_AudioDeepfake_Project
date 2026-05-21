# Baseline Model

## Summary

This repository contains a preliminary deep learning baseline for detecting
spoofed voice audio using the ASVspoof 2019 Logical Access dataset. The current
model is intentionally simple: it converts each audio clip into a log-mel
spectrogram and trains a small CNN to classify each sample as `bonafide` or
`spoof`.

The purpose of this baseline is to establish a stable, reproducible experiment
pipeline before moving to stronger architectures such as AASIST, Wav2Vec2, or
ensemble models. The pipeline now supports end-to-end training, checkpointing,
evaluation, prediction export, and error analysis.

Implemented so far:

- ASVspoof-style protocol parsing for `bonafide` / `spoof` labels
- robust audio loading with `torchaudio`, plus `soundfile` fallback for HPC
- 16 kHz mono preprocessing
- fixed-length 4 second clips with random crop during training and center crop
  during evaluation
- repeat padding for clips shorter than 4 seconds
- log-mel spectrogram feature extraction
- compact CNN binary classifier
- class-weighted binary cross-entropy training
- terminal logging after each epoch with train loss, dev loss, accuracy,
  ROC-AUC, EER, FPR, and FNR
- `best.pt` and `latest.pt` PyTorch checkpoints
- JSON/CSV experiment artifacts
- spectrogram visualization script
- prediction/error analysis script

## Feature Setup

This baseline model learns from log-mel spectrograms, which represent how speech energy is
distributed across frequency bands over time. In this representation, the
x-axis is time, the y-axis is mel-scaled frequency, and each value is log-scaled
energy. The CNN learns local spectro-temporal patterns that may distinguish real
human speech from synthetic or manipulated speech, such as harmonic texture,
high-frequency artifacts, smoothness, noise patterns, and unnatural transitions.

Current feature configuration:

| Setting | Value |
| --- | --- |
| Input audio | ASVspoof `.flac` utterances |
| Label task | Binary classification: `bonafide = 0`, `spoof = 1` |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Clip length | 4.0 seconds |
| Samples per clip | 64,000 |
| Training crop | Random 4-second crop |
| Evaluation crop | Center 4-second crop |
| Short-clip handling | Repeat pad to 4 seconds |
| Feature type | Log-mel spectrogram |
| Mel bands | 128 |
| FFT size | 1024 |
| Window length | 25 ms |
| Hop length | 10 ms |
| Frequency range | 20 Hz to 7,600 Hz |
| Feature normalization | Per-sample mean/std normalization |
| Approximate feature shape | `1 x 128 x 401` |

## Model Architecture

The baseline model is a small convolutional neural network:

```text
audio waveform
  -> log-mel spectrogram
  -> Conv2D + BatchNorm + GELU + MaxPool
  -> Conv2D + BatchNorm + GELU + MaxPool
  -> Conv2D + BatchNorm + GELU + MaxPool
  -> Conv2D + BatchNorm + GELU + MaxPool
  -> global average pooling
  -> dropout
  -> linear layer
  -> spoof logit
  -> sigmoid spoof score
```

The final score is interpreted as spoof risk. By default, scores greater than or
equal to `0.5` are predicted as `spoof`; scores below `0.5` are predicted as
`bonafide`.

## Expected ASVspoof Inputs

Uses ASVspoof 2019 LA train/dev data:

- train audio root, usually ending in `ASVspoof2019_LA_train/flac`
- train protocol, usually `ASVspoof2019.LA.cm.train.trn.txt`
- dev audio root, usually ending in `ASVspoof2019_LA_dev/flac`
- dev protocol, usually `ASVspoof2019.LA.cm.dev.trl.txt`

The parser expects ASVspoof protocol rows where the second token is the
utterance id and the final token is `bonafide` or `spoof`.

## Quick Sanity Checks

Visualize a few samples:

```bash
python scripts/visualize_samples.py \
  --config configs/baseline_logmel_cnn.yaml \
  --num-samples 8
```

To visualize dev samples instead of train samples:

```bash
python scripts/visualize_samples.py \
  --config configs/baseline_logmel_cnn.yaml \
  --split dev \
  --num-samples 8
```

Overfit a tiny subset first:

```bash
python scripts/train_baseline.py \
  --config configs/baseline_logmel_cnn.yaml \
  --limit-train 100 \
  --limit-dev 100 \
  --epochs 20 \
  --batch-size 16
```

To run a small baseline pass:

```bash
python scripts/train_baseline.py \
  --config configs/baseline_logmel_cnn.yaml \
  --limit-train 5000 \
  --limit-dev 5000 \
  --epochs 2
```

Evaluate a saved checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/runs/<run_id>/best.pt
```

By default, evaluation uses the dev paths from the checkpoint config. Use
`--split train` to evaluate the configured train split, or pass
`--audio-root` and `--protocol-path` to override the config.

Analyze prediction errors:

```bash
python scripts/analyze_predictions.py \
  --predictions outputs/runs/<run_id>/predictions_dev_best.csv \
  --protocol-path /path/to/ASVspoof2019.LA.cm.dev.trl.txt
```

The protocol path is optional. If provided, the report also groups metrics by
speaker ID and attack ID.

## Run Artifacts

Each training run writes to `outputs/runs/logmel_cnn_<timestamp>/`:

- `config.json`
- `history.json`
- `metrics_latest.json`
- `metrics_best.json`
- `predictions_dev_latest.csv`
- `predictions_dev_best.csv`
- `prediction_analysis.json`
- `prediction_analysis.txt`
- `latest.pt`
- `best.pt`

Reported metrics include accuracy, ROC-AUC, EER, FPR, FNR, and the confusion
matrix counts at the configured threshold.
