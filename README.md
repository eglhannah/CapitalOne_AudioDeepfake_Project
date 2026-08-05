# Detecting Audio Deepfakes for Fraud Prevention

**UVA MSDS Capstone × Capital One** — a machine learning system for detecting AI-generated and manipulated audio in fraud-relevant transactions, with a live inference dashboard, explainability layer, and reproducible model artifacts.

## Team

| Role | Name |
|---|---|
| Team member (AASIST modeling) | Arnav Jain |
| Team member (AWS deployment, log-mel CNN baseline) | Chase Cha |
| Team member (repo lead, explainability) | Hannah Egl |
| Team member (Wave2Vec 2.0 modeling, report lead) | Mohini Gupta |
| Faculty mentor | Daniel Graham |
| Sponsor, project manager | Mustufa Zaranwala (Capital One) |
| Sponsor | Arindam Chakraborty (Capital One) |
| Sponsor | Mehul Garnara (Capital One) |

## Headline results

50/50 score-level ensemble of AASIST v3 and Wave2Vec 2.0 across the three evaluation sets:

| Benchmark | AASIST v3 | Wave2Vec 2.0 | Ensemble (50/50) |
|---|---|---|---|
| ASVspoof 2019 LA eval | 1.67% EER | 1.25% EER | **1.02% EER** |
| ASVspoof 2021 LA eval | 4.67% EER | 4.49% EER | **3.18% EER** |
| ASVspoof 2021 DF eval | 17.01% EER | 16.38% EER | **14.87% EER** |

The ensemble clears the sponsor's ~15% EER target on 2021 DF and beats both standalone models on every benchmark. AASIST v3 (297K parameters) is the model deployed to the live dashboard due to its ~320× smaller footprint compared to Wave2Vec.

## Live demo

Chase's Lambda-hosted inference dashboard: **https://d1rd0z0qtd115u.cloudfront.net/**

Upload or record any audio clip (WAV, FLAC, MP3, M4A, OGG, WebM; up to 4 MiB). The dashboard returns a spoof score, threshold, prediction, and metadata. Audio is discarded immediately after inference and is not stored.

## Repository layout

```
aasist/                     Arnav's AASIST modeling branch (v1, v2, v3)
├── code/                   training scripts, sbatch files, eval scripts
├── results/                EER metrics, comparison charts, ROC curves
│   └── v3_predictions/     per-utterance predictions on all eval sets
├── simple_aasist.py        minimal loader wrapping HuggingFace + inference
├── handoff/                integration notes for the deployment + SHAP teams
└── README.md               branch-specific docs

w2v/                        Mohini's Wave2Vec 2.0 modeling branch
├── no augmentation/        base w2v runs
├── rawboost/               RawBoost-augmented runs
├── simple_model.py         minimal inference loader
├── test_predictions_*.csv  per-utterance predictions
└── README.md

logmel_cnn_baseline/        Chase's log-mel CNN baseline
├── src/                    model + training code
├── scripts/                training and evaluation scripts
└── README.md

explainability/             Hannah's SHAP + LIME explainability workstream
├── interpret_aasist_shap.py         occlusion-based SHAP wrapper for AASIST
├── dash_aasist_app.py               local dashboard for interactive SHAP viewing
├── INTERPRET_AASIST_SHAP_GUIDE.md   walkthrough
├── SHAP_vs_LIME.md                  framework comparison
└── SHAP_Features_Justification.md   feature-selection rationale

deployment/aasist_lambda/   Chase's AWS Lambda deployment
└── vendor/aasist/models/   vendored AASIST model class

eval_keys/                  Canonical ASVspoof 2021 DF evaluation keys
├── ASVspoof2021_DF_keys.csv                 slim CSV (file_id, label, attack, compression)
├── ASVspoof2021_DF_trial_metadata.txt.gz    full original 13-column keys
└── README.md                                schema + join snippet

scripts/                    Shared cross-workstream utilities
└── HuggingFaceModel_Links.py    Central registry of HF model URLs

Notes & Meeting Plans/      Weekly meeting plans, demo videos, sponsor update audio
ASVspoof_Dataset_Overview.md    Dataset description used by all workstreams
```

## Trained models on HuggingFace

| Model | HuggingFace repo |
|---|---|
| AASIST v1 (baseline) | `arnavjain321/aasist-v1-baseline` |
| AASIST v2 (RawBoost augmentation) | `arnavjain321/aasist-v2-rawboost` |
| AASIST v3 (real codec augmentation, deployed) | `arnavjain321/aasist-v3-codecaugment` |
| Wave2Vec 2.0 (no augmentation, 4s) | `rde6mn/no_aug_w2v_4s` |
| Log-mel CNN baseline | `chasecha/logmel_cnn_baseline` |

## Quickstart — score an audio file with AASIST v3

```python
import torch, soundfile as sf
from aasist.simple_aasist import load_aasist_v3, predict

model = load_aasist_v3()

waveform, sr = sf.read("your_audio.flac")
waveform = torch.from_numpy(waveform).float().unsqueeze(0)  # shape (1, N)

out = predict(model, waveform)
print(f"spoof probability: {out['spoof_prob'].item():.4f}")
```

`out` is a dict with `embedding`, `logits`, and `spoof_prob` (index 1 of softmax).

## Data

Training and evaluation use the ASVspoof challenge datasets. See `ASVspoof_Dataset_Overview.md` for a full description.

- **2019 LA**: Datashare Edinburgh, https://datashare.ed.ac.uk/handle/10283/3336
- **2021 LA / DF**: Zenodo records 4837263 (LA) and 4835108 (DF, 4 parts)

Canonical 2021 DF evaluation keys are mirrored in `eval_keys/` for reproducible scoring across the team.

## Reproducibility

- All AASIST results are reproducible from the code in `aasist/code/` with the fixed random seed and cuDNN deterministic flag documented in `aasist/results/reproducibility.md`
- Wave2Vec 2.0 runs are documented in the notebooks under `w2v/no augmentation/` and `w2v/rawboost/`
- The AWS Lambda inference has been verified byte-for-byte identical to the local evaluation pipeline (see `aasist/results/v3_predictions/` for local scores that match the deployed dashboard)
