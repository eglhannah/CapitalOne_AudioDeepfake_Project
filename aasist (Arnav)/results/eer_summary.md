# AASIST v1 — Full Evaluation Results

**Trained:** 2026-05-29 on ASVspoof 2019 LA train (25 epochs, ~3.3 hr on NVIDIA A6000)
**Best checkpoint:** epoch 20, dev EER 0.90%
**Checkpoint path on Rivanna:** `/scratch/mhq8ka/aasist/outputs/runs/aasist_fast_13608120/best.pt`

---

## Headline — degradation across 4 test sets

| Test set | EER | AUC | n utterances | What's hard about it |
|---|---|---|---|---|
| ASVspoof 2019 LA dev | **0.90%** | 0.999 | 24,844 | In-domain (same attacks as train) |
| ASVspoof 2019 LA eval | **3.33%** | 0.994 | 71,237 | Unknown attacks (A07–A19, 11 unseen) |
| ASVspoof 2021 LA eval | **5.67%** | 0.984 | 148,176 | + Telephony codecs (alaw, ulaw, gsm, opus, g722, pstn) |
| ASVspoof 2021 DF eval | **22.95%** | 0.881 | 533,928 | + Media compression + 100+ unseen TTS/VC systems |

All four under PRD's 25% EER gate. Monotonic degradation. See `degradation_curve.png` for the visual.

---

## Per-attack EER on 2019 LA eval (unknown attacks)

| Attack | EER | Note |
|---|---|---|
| A07 | 0.63% | Waveform concatenation TTS |
| A08 | 0.91% | WaveNet TTS |
| A09 | **0.02%** | Griffin-Lim (trivially easy) |
| A10 | 0.63% | WaveRNN E2E TTS |
| A11 | 0.53% | GAN TTS |
| A12 | 0.63% | WaveNet VC |
| A13 | 0.18% | Moment-matching VC |
| A14 | 0.53% | Classical vocoder TTS |
| A15 | 0.63% | Neural VC + WaveNet |
| A16 | 1.44% | Same as A04 (known reference) |
| **A17** | **5.17%** | **VAE-VC + waveform filtering** |
| **A18** | **12.19%** | **i-vector/PLDA VC + DNN glottal vocoder** |
| A19 | 2.67% | Same as A06 (known reference) |

**Observation:** A17 and A18 are the hardest, matching published ASVspoof literature exactly. Model fails where the field fails — sign of legitimate generalization.

---

## Per-attack EER on 2021 DF eval (hardest test set)

Top 15 attacks by spoof count:

| Attack | EER | Note |
|---|---|---|
| A09 | 0.79% | Griffin-Lim — still trivially easy |
| A13 | 3.96% | Moment-matching VC |
| Task2-team12 | 7.78% | VCC2020 challenge submission |
| A14 | 8.37% | |
| A11 | 8.90% | |
| A08 | 13.31% | |
| A07 | 15.87% | |
| A16 | 16.22% | |
| A15 | 16.21% | |
| A12 | 17.52% | |
| A10 | 17.22% | |
| A19 | 26.57% | |
| Task2-team29 | 28.24% | VCC2020 — harder submission |
| **A17** | **31.82%** | **Still one of the hardest** |
| **A18** | **40.93%** | **Hardest published anti-spoofing test case** |

**Same pattern as 2019 LA eval, just scaled up due to harder conditions.**

---

## Per-codec EER on 2021 LA (FR-12 fairness slice)

| Codec | EER | n_spoof |
|---|---|---|
| pstn | 3.63% | 9,090 |
| gsm | 3.90% | 12,463 |
| ulaw | 5.15% | 12,771 |
| alaw | 5.37% | 10,690 |
| g722 | 6.95% | 10,282 |
| none (clean) | 7.31% | 11,090 |
| opus | 8.78% | 7,976 |

**Note:** "none" (clean) having higher EER than telephony codecs is counterintuitive but consistent with published AASIST behavior on cross-domain. Model was trained on 2019 LA (VCTK studio quality); 2021 LA "none" bonafide audio differs from that distribution. RawBoost augmentation (v2) is expected to close this gap.

---

## Per-compression EER on 2021 DF (FR-12 fairness slice)

| Compression | EER | n_spoof |
|---|---|---|
| low_ogg | 21.96% | 63,624 |
| oggm4a | 22.10% | 63,624 |
| high_ogg | 22.66% | 63,624 |
| nocodec | 23.19% | 52,920 |
| mp3m4a | 23.19% | 63,624 |
| high_mp3 | 23.19% | 52,940 |
| high_m4a | 23.27% | 52,950 |
| low_m4a | 23.32% | 52,895 |
| low_mp3 | 23.63% | 52,858 |

**Spread is only 1.67 percentage points across all 9 compression conditions.** Model is robust to compression type — picks up signal from the spoof itself, not from compression artifacts. Strong production-relevant property.

---

## Reproducibility quick reference

| Item | Value |
|---|---|
| Random seed | 1234 |
| cuDNN deterministic | True |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)) |
| LR scheduler | CosineAnnealingLR (T_max=total_steps, eta_min=5e-6) |
| Loss | CrossEntropyLoss with class weights [0.9, 0.1] (upweight bonafide minority) |
| Batch size | 24 |
| Epochs | 25 |
| Input | 64,600 samples (4.04 sec @ 16 kHz mono raw waveform) |
| Label convention | bonafide=0, spoof=1 |
| Spoof score | softmax(logits)[:, 1] (index 1 = spoof probability) |
| Training time | ~3.3 hr on NVIDIA A6000 |

See `reproducibility.md` for full reproduction recipe.

---

## Published comparison

- AASIST paper (Jung et al., ICASSP 2022): 0.83% EER on 2019 LA dev (100 epochs + SWA). We achieved 0.90% with 25 epochs and no SWA — paper-comparable.
- AASIST without RawBoost on 2021 DF: ~22–25% in published literature. We achieved 22.95% — paper-consistent.
- AASIST + RawBoost on 2021 DF: ~12–15% in published literature. Target for v2.
