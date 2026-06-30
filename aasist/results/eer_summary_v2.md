# AASIST v2 (RawBoost) Full Evaluation Results

**Trained:** 2026-06-09 on ASVspoof 2019 LA train (25 epochs, ~3 hr on NVIDIA A6000), RawBoost data augmentation enabled
**Best checkpoint:** epoch 25, dev EER 0.78%
**Checkpoint path on Rivanna:** `/scratch/mhq8ka/aasist/outputs/runs/aasist_v2_rawboost_14642481/best.pt`

---

## Headline: v1 vs v2 across all 4 test sets

| Dataset | v1 EER | v2 EER | Delta | Direction |
|---|---|---|---|---|
| 2019 LA dev (in domain, seen attacks) | 0.90% | **0.78%** | -0.12 pp | Better |
| 2019 LA eval (unknown attacks A07-A19) | 3.33% | **1.87%** | -1.46 pp | Better (-44% relative) |
| 2021 LA eval (telephony codecs) | 5.67% | 8.01% | +2.34 pp | Worse (regression) |
| **2021 DF eval (compression + 100+ TTS/VC)** | **22.95%** | **17.20%** | **-5.75 pp** | **Better (-25% relative)** |

All four below PRD's 25% gate. v2 improves on 3 of 4 datasets.

See `aasist_v1_vs_v2_degradation_curve.png` for the visual.

---

## Per-attack EER on 2021 DF (sorted by v1 difficulty)

The hardest published attacks (A17, A18) improved most under RawBoost:

| Attack | v1 EER | v2 EER | Improvement |
|---|---|---|---|
| A18 (i-vector/PLDA VC + DNN glottal vocoder) | 40.93% | **27.38%** | **-13.55 pp** |
| A17 (VAE-VC + waveform filtering) | 31.82% | **10.13%** | **-21.69 pp** |
| Task2-team29 (VCC2020) | 28.24% | 27.39% | -0.85 pp |
| A19 | 26.57% | 6.29% | -20.28 pp |
| A12 | 17.52% | 18.27% | +0.75 pp |
| A10 | 17.22% | 20.96% | +3.74 pp |
| A15 | 16.21% | 17.12% | +0.91 pp |
| A16 | 16.22% | 8.14% | -8.08 pp |
| A07 | 15.87% | 15.69% | -0.18 pp |
| A08 | 13.31% | 8.26% | -5.05 pp |
| A11 | 8.90% | 11.49% | +2.59 pp |
| A14 | 8.37% | 4.41% | -3.96 pp |
| Task2-team12 (VCC2020) | 7.78% | 2.63% | -5.15 pp |
| A13 | 3.96% | 4.24% | +0.28 pp |
| A09 (Griffin-Lim) | 0.79% | 1.17% | +0.38 pp |

---

## Per-compression EER on 2021 DF (FR-12 fairness slice)

All 9 compression conditions improved:

| Compression | v1 EER | v2 EER | Delta |
|---|---|---|---|
| low_ogg | 21.96% | 15.48% | -6.48 pp |
| oggm4a | 22.10% | 15.52% | -6.58 pp |
| high_ogg | 22.66% | 17.34% | -5.32 pp |
| nocodec | 23.19% | 18.03% | -5.16 pp |
| mp3m4a | 23.19% | 17.28% | -5.91 pp |
| high_mp3 | 23.19% | 18.06% | -5.13 pp |
| high_m4a | 23.27% | 18.03% | -5.24 pp |
| low_m4a | 23.32% | 17.69% | -5.63 pp |
| low_mp3 | 23.63% | 17.81% | -5.82 pp |

v2 spread: 2.55 pp (15.48% to 18.03%). v1 spread was 1.67 pp. Slightly less uniform but robustness story preserved.

---

## Per-codec EER on 2021 LA (the regression)

| Codec | v1 EER | v2 EER | Direction |
|---|---|---|---|
| gsm | 4.10% | 3.97% | Better |
| pstn | 3.63% | 3.72% | Flat |
| alaw | 5.37% | 8.03% | Worse |
| g722 | 7.49% | 9.74% | Worse |
| ulaw | 5.35% | 8.00% | Worse |
| opus | 6.62% | 8.59% | Worse |
| none (clean) | 7.65% | 10.06% | Worse |

Hypothesis: RawBoost's codec simulation (mu-law / A-law companding + 8 bit quantization) does not exactly match real telephony codec transmission as used in 2021 LA (Asterisk PBX with real codec round trips). GSM and PSTN, the codecs least similar to RawBoost's simulation, improved or stayed flat.

Next sprint: codec realistic augmentation (ffmpeg actual codec round trips) to address.

---

## Latency profiling (Mustafa Rec #3)

Sliding window inference: 4 second window with 50% overlap (2 second stride). Audio shorter than 4 sec is padded; audio longer than 4 sec is windowed, scores averaged.

### GPU (NVIDIA A6000)

| Clip | Windows | Mean ms | p95 ms | Throughput |
|---|---|---|---|---|
| 1 sec | 1 | 7.92 | 7.98 | 126 clips/sec |
| 4 sec | 1 | 7.95 | 8.02 | 126 clips/sec |
| 5 sec | 2 | 15.80 | 15.89 | 63 clips/sec |
| 10 sec | 4 | 31.56 | 31.65 | 32 clips/sec |
| 30 sec | 14 | 110.14 | 110.77 | 9 clips/sec |

Worst case p95: 111 ms. **2167 times under PRD's 240,000 ms target.**

### CPU (no GPU, production scenario)

| Clip | Windows | Mean ms | p95 ms | Throughput |
|---|---|---|---|---|
| 1 sec | 1 | 198 | 199 | 5 clips/sec |
| 4 sec | 1 | 201 | 204 | 5 clips/sec |
| 5 sec | 2 | 396 | 397 | 2.5 clips/sec |
| 10 sec | 4 | 792 | 795 | 1.3 clips/sec |
| 30 sec | 14 | 2,771 | 2,777 | 0.4 clips/sec |

Worst case p95: 2.8 seconds. **86 times under PRD target.**

---

## Reproducibility

Same setup as v1, with RawBoost augmentation enabled. See `reproducibility.md` for full recipe.

| Item | Value |
|---|---|
| Random seed | 1234 |
| cuDNN deterministic | True |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| LR schedule | CosineAnnealingLR (eta_min=5e-6) |
| Loss | CrossEntropyLoss with class weights [0.9, 0.1] |
| Batch size | 24 |
| Epochs | 25 |
| RawBoost probabilities | p_lnl=0.5, p_isd=0.5, p_ssi=0.5, p_codec=0.5 |
| Training time | ~3 hr on NVIDIA A6000 |
| SLURM allocation | ds2022 |

---

## Files

- **Checkpoint:** `/scratch/mhq8ka/aasist/outputs/runs/aasist_v2_rawboost_14642481/best.pt`
- **Eval results:** `/scratch/mhq8ka/aasist/outputs/eval/v2_2019_LA_dev/`, `v2_2019_LA_eval/`, `v2_2021_LA_eval/`, `v2_2021_DF_eval/`
- **Latency JSONs:** `/scratch/mhq8ka/aasist/outputs/latency/v2_gpu.json`, `v2_cpu.json`
- **Training script:** `code/train_aasist_v2.py`
- **RawBoost module:** `code/rawboost.py`
- **Eval wrappers:** `code/run_eval_v2_*.sh`, `code/sbatch_eval_v2_*.sh`
- **Latency profiler:** `code/latency_profile.py`, `code/sbatch_latency_profile.sh`
