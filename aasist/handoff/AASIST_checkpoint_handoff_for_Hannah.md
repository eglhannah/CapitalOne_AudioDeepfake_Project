# AASIST checkpoint handoff — for Hannah

Hey — here's everything you need to load and run my trained AASIST model.

## Where the checkpoint lives

**Path on Rivanna:**
```
/scratch/mhq8ka/aasist/outputs/runs/aasist_fast_13608120/
```

Files in that directory:
- `best.pt` — the checkpoint with lowest dev EER (epoch 20, dev EER 0.90%)
- `latest.pt` — the last-epoch checkpoint (epoch 25, but dev EER spiked here, use `best.pt`)
- `config.json` — the exact hyperparameters used for training
- `history.json` — per-epoch metrics for the full 25-epoch training run

If you can't read directly from `/scratch/mhq8ka/`, let me know — I'll either chmod it or copy to a shared location.

## Quick model context

- **Architecture:** AASIST from clovaai (https://github.com/clovaai/aasist)
- **Trained on:** ASVspoof 2019 LA train split (~25,380 utterances)
- **Best dev EER:** 0.90% on ASVspoof 2019 LA dev (~24,844 utterances)
- **Cross-domain results I've already run:**
  - 2019 LA eval (unknown attacks): 3.33%
  - 2021 LA eval (telephony codecs): 5.67%
  - 2021 DF eval (media compression + 100+ unseen TTS/VC): 22.95%
- **Model size:** ~297K parameters (tiny — runs in <100 MB memory, <5 ms inference on A100)

## Conventions (important — these match Chase's pipeline)

- **Label:** bonafide = 0, spoof = 1
- **Spoof score:** after softmax over the 2 output logits, **take index 1**. Higher value = more likely spoof.
- **Input audio:** raw waveform, 16 kHz mono, **64,600 samples** (~4.04 seconds). Crop or repeat-pad to that exact length.

## How to load the model and score audio

```python
import sys
import torch
import numpy as np

# Add the AASIST repo to path so we can import the model class
sys.path.insert(0, "/scratch/mhq8ka/aasist/code/aasist")
from models.AASIST import Model as AASISTModel

# Model config (must match training)
AASIST_CFG = {
    "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

# Load the model + checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(
    "/scratch/mhq8ka/aasist/outputs/runs/aasist_fast_13608120/best.pt",
    map_location=device, weights_only=False,
)
model = AASISTModel(AASIST_CFG).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# Score one audio clip (must be exactly 64600 samples)
@torch.no_grad()
def get_spoof_score(waveform_np):
    """waveform_np: numpy float32 array of length 64600 (16 kHz mono)."""
    x = torch.from_numpy(waveform_np).float().unsqueeze(0).to(device)
    _, logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, 1])  # index 1 = spoof prob
```

## How to load audio so it has the right shape

I use Chase's dataloader for this (it handles FLAC, resampling, cropping, padding):

```python
import sys
sys.path.insert(0, "/scratch/mhq8ka/aasist/code/CapitalOne_AudioDeepfake_Project/logmel_cnn_baseline/src")
from asv_baseline.data.asvspoof_dataset import ASVspoofLADataset, parse_la_protocol

# Or, if you want to skip the Dataset class and load one file manually:
import soundfile as sf
wav, sr = sf.read("path/to/audio.flac", dtype="float32")
# resample to 16000 if needed; pad/crop to 64600 samples
```

If you want to batch-process a lot of audio, copy my eval scripts from:
```
/scratch/mhq8ka/aasist/code/aasist_branch/
  eval_aasist.py            # for 2019 LA
  eval_aasist_2021_la.py    # for 2021 LA
  eval_aasist_2021_df.py    # for 2021 DF
```

Each of those takes `--checkpoint` and a protocol/metadata file and writes per-attack/per-codec/per-compression EER breakdowns. Easy to fork for other slicing.

## ⚠️ One pitfall to avoid

The 2021 LA/DF audio has some FLAC variants that crash `libsndfile`. My loader uses a 3-tier fallback (torchaudio → soundfile → **ffmpeg subprocess**). If you're loading 2021 audio, install ffmpeg in your conda env:

```bash
conda install -n <your_env> -c conda-forge ffmpeg -y
```

(System ffmpeg at `/usr/bin/ffmpeg` is broken on Rivanna — missing `libvmaf.so.0`. Use conda-forge's instead.)

## What I've already computed (if you want the raw output)

Per-utterance predictions are saved as CSVs:
```
/scratch/mhq8ka/aasist/outputs/eval/2019_LA_eval/predictions.csv
/scratch/mhq8ka/aasist/outputs/eval/2021_LA_eval/predictions.csv
/scratch/mhq8ka/aasist/outputs/eval/2021_DF_eval/predictions.csv
```

Each is `utterance_id, label (0/1), score (spoof prob 0-1)`. Already great for FR-12 fairness slicing — you can join against the metadata to slice by codec, attack, speaker, etc.

## Reproducibility / context

- **Trained with:** seed=1234, cudnn.deterministic=True, 25 epochs, Adam optimizer (lr=1e-4, wd=1e-4), cosine LR schedule, CrossEntropyLoss with class weights `[0.9, 0.1]` upweighting bonafide minority
- **Training script:** `/scratch/mhq8ka/aasist/code/aasist_branch/train_aasist.py`
- **Wall time:** 3.3 hr on NVIDIA A6000
- **Published reference:** Jung et al., AASIST, ICASSP 2022 (arXiv:2110.01200) — reports 0.83% EER on this dataset, so we're paper-comparable

Yell at me on Slack if anything's broken or unclear.
