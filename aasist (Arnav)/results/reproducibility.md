# Reproducibility — AASIST v1 Training and Evaluation

This document gives exact instructions to reproduce the v1 numbers reported in `eer_summary.md`. Everything is on Rivanna and tied to a fixed random seed (1234) with `cudnn.deterministic=True`.

---

## Environment setup (one-time)

```bash
# Load miniforge
module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh

# Create env (or activate existing if you've already built it)
conda config --add pkgs_dirs ~/.conda/pkgs
mkdir -p ~/.conda/pkgs ~/.conda/envs
conda create -n aasist python=3.11 -y
conda activate aasist

# Install dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy tqdm tensorboard soundfile librosa pyyaml

# IMPORTANT: install ffmpeg via conda-forge — system /usr/bin/ffmpeg is broken on Rivanna
conda install -c conda-forge ffmpeg -y
```

After each fresh shell, re-activate:

```bash
module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh
conda deactivate  # defensive — fixes a known PATH bug
conda activate aasist
which python  # MUST show ~/.conda/envs/aasist/bin/python (not /apps/...)
```

---

## Data setup (one-time)

```bash
mkdir -p /scratch/$USER/aasist/data
cd /scratch/$USER/aasist/data

# 2019 LA (train + dev + eval together, ~7.2 GB compressed)
wget -c "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip" -O LA.zip
unzip -q LA.zip

# 2021 LA + DF eval (~48 GB total) — see code/download_2021.sh
bash /scratch/$USER/aasist/code/aasist_branch/download_2021.sh
# Then extract:
cd /scratch/$USER/aasist/data/2021
tar -xzf ASVspoof2021_LA_eval.tar.gz
tar -xzf LA-keys-full.tar.gz
for i in 00 01 02 03; do tar -xzf ASVspoof2021_DF_eval_part${i}.tar.gz; done
tar -xzf DF-keys-full.tar.gz
```

Final directory structure on `/scratch/$USER/aasist/data/`:

```
LA/
  ASVspoof2019_LA_train/flac/    # train audio
  ASVspoof2019_LA_dev/flac/      # dev audio
  ASVspoof2019_LA_eval/flac/     # 2019 eval audio
  ASVspoof2019_LA_cm_protocols/  # 2019 protocols
2021/
  ASVspoof2021_LA_eval/flac/     # 2021 LA audio
  ASVspoof2021_DF_eval/flac/     # 2021 DF audio
  keys/LA/CM/trial_metadata.txt  # 2021 LA labels
  keys/DF/CM/trial_metadata.txt  # 2021 DF labels
```

---

## Code setup (one-time)

```bash
mkdir -p /scratch/$USER/aasist/code
cd /scratch/$USER/aasist/code

# Clone AASIST reference impl
git clone https://github.com/clovaai/aasist.git

# Clone team repo (this folder lives in there)
git clone https://github.com/eglhannah/CapitalOne_AudioDeepfake_Project.git

# Create branch workspace and copy AASIST scripts from this folder
mkdir -p aasist_branch
cp "CapitalOne_AudioDeepfake_Project/aasist (Arnav)/code/"*.py aasist_branch/
cp "CapitalOne_AudioDeepfake_Project/aasist (Arnav)/code/"*.sh aasist_branch/
chmod +x aasist_branch/*.sh
```

---

## Reproduce v1 training (the 0.90% dev EER result)

```bash
sbatch /scratch/$USER/aasist/code/aasist_branch/sbatch_aasist_fast.sh
```

Job runs ~3.3 hr on the A6000 partition. Output lands at:

```
/scratch/$USER/aasist/outputs/runs/aasist_fast_<JOBID>/
├── best.pt            # checkpoint with lowest dev EER
├── latest.pt          # last-epoch checkpoint
├── history.json       # per-epoch metrics
└── config.json        # hyperparameters
```

**Expected:** Best dev EER ~0.85–0.95% at epoch ~18–22 (the exact best epoch varies slightly across runs even with fixed seed due to non-deterministic CUDA kernels on some operations, but the best EER reliably lands in this range).

---

## Reproduce v1 evaluations

After training (or pointing at the canonical checkpoint at `/scratch/mhq8ka/aasist/outputs/runs/aasist_fast_13608120/best.pt`):

```bash
# 2019 LA eval (~5 min on A100)
srun --jobid=$SLURM_JOB_ID bash /scratch/$USER/aasist/code/aasist_branch/run_eval_2019.sh

# 2021 LA eval (~25 min on A100)
srun --jobid=$SLURM_JOB_ID bash /scratch/$USER/aasist/code/aasist_branch/run_eval_2021_la.sh

# 2021 DF eval (~80 min on A100, fire-and-forget via sbatch)
sbatch /scratch/$USER/aasist/code/aasist_branch/sbatch_eval_2021_df.sh
```

Each eval writes:
- `eval_summary.json` — overall + per-attack + per-codec (or per-compression for DF) breakdowns
- `predictions.csv` — per-utterance `(utterance_id, label, score)` triples
- `skipped_files.csv` — bad audio files if any (should be 0 with ffmpeg installed correctly)

---

## Reproduce the plots

After all four evals complete:

```bash
python "CapitalOne_AudioDeepfake_Project/aasist (Arnav)/code/plot_degradation_curve.py"  # ./aasist_degradation_curve.png
python "CapitalOne_AudioDeepfake_Project/aasist (Arnav)/code/plot_eer_curve.py"          # ./aasist_training_curve.png
```

Both plots have the EER numbers and history inlined as Python constants — they don't read from JSON files. If numbers change, edit the constants at the top of each script.

---

## Hyperparameters (exact values used for the reported 0.90% result)

| Setting | Value | Source |
|---|---|---|
| Random seed | 1234 | `train_aasist.py` `seed_all(1234)` |
| cuDNN deterministic | True | `train_aasist.py` `torch.backends.cudnn.deterministic = True` |
| cuDNN benchmark | False | `train_aasist.py` `torch.backends.cudnn.benchmark = False` |
| Epochs | 25 | `sbatch_aasist_fast.sh` `--epochs 25` |
| Batch size | 24 | AASIST paper default |
| Learning rate | 1e-4 | AASIST paper default |
| Weight decay | 1e-4 | AASIST paper default |
| Optimizer | Adam | `betas=(0.9, 0.999), amsgrad=False` |
| LR schedule | CosineAnnealingLR | `T_max = epochs × steps_per_epoch`, `eta_min = 5e-6` |
| Loss | CrossEntropyLoss | weights `[0.9, 0.1]` (upweight bonafide minority) |
| Label convention | bonafide=0, spoof=1 | Chase's convention; opposite of AASIST paper |
| Spoof score | `softmax(logits, dim=-1)[:, 1]` | Index 1 = spoof prob |
| Input length | 64,600 samples | AASIST paper default (4.04 sec @ 16 kHz) |
| Sample rate | 16,000 Hz | Standard |
| Channels | 1 (mono) | Chase's dataloader converts |

---

## Reference architecture (AASIST model config)

Exact `AASIST_CFG` dict used:

```python
AASIST_CFG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
```

This matches the published AASIST paper configuration exactly. The model is `Model` class from `clovaai/aasist/models/AASIST.py`, ~297K parameters.

---

## Known gotchas

| Gotcha | Symptom | Fix |
|---|---|---|
| Broken system ffmpeg | `libsndfile` errors on 2021 LA/DF FLAC files | `conda install -c conda-forge ffmpeg -y` |
| `conda activate` doesn't fully activate | `which python` shows `/apps/...` not `~/.conda/...` | `conda deactivate` then re-activate |
| OOD shell paste line-break | Long commands split across lines | Use `sbatch` for long commands, or short single-line commands in shell |
| `np.trapz` removed in NumPy 2.x | `AttributeError: module 'numpy' has no attribute 'trapz'` in Chase's `metrics.py` | Apply `sed -i 's/np\.trapz/np.trapezoid/g' .../metrics.py` |
| Slurm allocation name | `salloc: PrologSlurmctld failed, job killed` | Use `-A ds6015` (capstone allocation, Daniel Graham). Personal fallback: `mhq8ka_rivanna`. |

---

## Contact

Questions about the AASIST workstream — Arnav Jain (arnav.jain321@gmail.com).
