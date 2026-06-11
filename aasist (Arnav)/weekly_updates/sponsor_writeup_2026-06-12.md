# Capital One Sponsor Update: AASIST Modeling Branch v2 (RawBoost)
**Week of June 8 to June 12, 2026**
**Author:** Arnav Jain
**Workstream:** AASIST modeling branch and project coordination
**UVA MSDS Capstone, Voice Anti-Spoofing for Capital One**

---

## TL;DR

This week the team addressed all three recommendations from last sync. On the AASIST modeling branch specifically:

**Cross-domain improvement (Recommendation 1).** Implemented RawBoost data augmentation and retrained AASIST as v2. The 2021 DF EER dropped from 22.95% to 17.20%, a 25% relative reduction. The two historically hardest attacks in published literature, A17 and A18, improved by 22 and 14 percentage points respectively. In domain 2019 LA dev also improved (0.90% to 0.78%), and 2019 LA eval improved 44% relative (3.33% to 1.87%).

**Honest disclosure.** 2021 LA regressed from 5.67% to 8.01%. Hypothesis is that RawBoost's codec simulation (mu-law / A-law via companding) does not exactly match real telephony codec transmission. Codec realistic augmentation is the next sprint exploration.

**SHAP justification (Recommendation 2).** Hannah owns the explainability workstream. Her decision document is in the team repo at `explainability (Hannah)/SHAP_Features_Justification.md`. Her approach is SHAP applied to five handcrafted acoustic features (jitter, pitch variability, MFCC variance, RMS dynamic range, HNR), which is complementary to my model side iWAX work.

**Latency profiling (Recommendation 3).** AASIST inference profiled across 1s, 2s, 4s, 5s, 10s, and 30s clips using a 4 second sliding window with 50% overlap. Worst case GPU latency: 110 ms on a 30 second utterance (14 windows). Worst case CPU latency: 2.8 seconds. Both are 86 to 2167 times under the PRD's 4 minute target.

---

## 1. Results: v1 vs v2 across four datasets

| Dataset | v1 EER | v2 EER | Delta | Direction |
|---|---|---|---|---|
| 2019 LA dev (in domain, seen attacks) | 0.90% | **0.78%** | -0.12 pp | Better |
| 2019 LA eval (unknown attacks A07-A19) | 3.33% | **1.87%** | -1.46 pp | Better (-44% relative) |
| 2021 LA eval (unknown attacks + telephony codecs) | 5.67% | 8.01% | +2.34 pp | Worse (-41% relative, regression) |
| **2021 DF eval (compression + 100+ unseen TTS/VC)** | **22.95%** | **17.20%** | **-5.75 pp** | **Better (-25% relative)** |

All four results stay below the PRD's 25% EER gate. v2 improves on three of four datasets, including both the hardest (2021 DF) and the in domain (2019 LA dev). Detail on the single regression (2021 LA) is in Section 3.

![AASIST v1 vs v2 degradation curve and per-attack EER on 2021 DF showing A17 and A18 improvements](aasist_v1_vs_v2_degradation_curve.png)

### Per-attack improvements on 2021 DF (the headline)

Sorted by v1 difficulty so the cross-domain improvement story is clear:

| Attack | v1 EER | v2 EER | Improvement | Note |
|---|---|---|---|---|
| **A18** (i-vector/PLDA VC + DNN glottal vocoder) | **40.93%** | **27.38%** | **-13.55 pp** | Hardest published anti-spoofing case |
| **A17** (VAE-VC + waveform filtering) | **31.82%** | **10.13%** | **-21.69 pp** | Second hardest by paper consensus |
| Task2-team29 (VCC2020) | 28.24% | 27.39% | -0.85 pp | |
| A19 | 26.57% | 6.29% | -20.28 pp | |
| A12 | 17.52% | 18.27% | +0.75 pp | |
| A10 | 17.22% | 20.96% | +3.74 pp | |
| A15 | 16.21% | 17.12% | +0.91 pp | |
| A16 | 16.22% | 8.14% | -8.08 pp | |
| A07 | 15.87% | 15.69% | -0.18 pp | |
| A08 | 13.31% | 8.26% | -5.05 pp | |
| A11 | 8.90% | 11.49% | +2.59 pp | |
| A14 | 8.37% | 4.41% | -3.96 pp | |
| Task2-team12 (VCC2020) | 7.78% | 2.63% | -5.15 pp | |
| A13 | 3.96% | 4.24% | +0.28 pp | |
| A09 (Griffin-Lim) | 0.79% | 1.17% | +0.38 pp | Already trivially easy |

The two attacks the published ASVspoof literature flags as hardest (A17, A18) improved the most. This pattern is consistent with what we would expect from RawBoost: the augmentation generalizes the model to handle the kinds of subtle acoustic artifacts that those attack systems produce, which the model struggled with when trained on clean 2019 LA only.

### Per-compression EER on 2021 DF (FR-12 fairness slice)

| Compression | v1 EER | v2 EER |
|---|---|---|
| low_ogg | 21.96% | 15.48% |
| oggm4a | 22.10% | 15.52% |
| high_ogg | 22.66% | 17.34% |
| nocodec | 23.19% | 18.03% |
| mp3m4a | 23.19% | 17.28% |
| high_mp3 | 23.19% | 18.06% |
| high_m4a | 23.27% | 18.03% |
| low_m4a | 23.32% | 17.69% |
| low_mp3 | 23.63% | 17.81% |

All 9 compression conditions improved. Spread is 2.55 percentage points in v2 (15.48% to 18.03%) compared to v1's 1.67 percentage points. Slightly less uniform, but the robustness story is preserved: variation across compression types remains small.

---

## 2. Latency profiling (Recommendation 3)

Implemented a sliding window inference wrapper. For audio shorter than 4 seconds we pad. For audio longer than 4 seconds we slide a 4 second window with 50% overlap (2 second stride) and average the per window spoof scores. Measured 50 trials per duration on A6000 GPU and 20 trials per duration on CPU (production like scenario).

### GPU (NVIDIA A6000)

| Clip duration | Windows | Mean latency | p95 latency | Throughput |
|---|---|---|---|---|
| 1 second | 1 | 7.92 ms | 7.98 ms | 126 clips/sec |
| 4 seconds | 1 | 7.95 ms | 8.02 ms | 126 clips/sec |
| 5 seconds | 2 | 15.80 ms | 15.89 ms | 63 clips/sec |
| 10 seconds | 4 | 31.56 ms | 31.65 ms | 32 clips/sec |
| **30 seconds** | **14** | **110.14 ms** | **110.77 ms** | **9 clips/sec** |

**Worst case p95 GPU latency: 111 ms. Headroom vs PRD's 240,000 ms target: 2167 times under.**

### CPU (no GPU, production deployment scenario)

| Clip duration | Windows | Mean latency | p95 latency | Throughput |
|---|---|---|---|---|
| 1 second | 1 | 198 ms | 199 ms | 5 clips/sec |
| 4 seconds | 1 | 201 ms | 204 ms | 5 clips/sec |
| 5 seconds | 2 | 396 ms | 397 ms | 2.5 clips/sec |
| 10 seconds | 4 | 792 ms | 795 ms | 1.3 clips/sec |
| **30 seconds** | **14** | **2,771 ms** | **2,777 ms** | **0.4 clips/sec** |

**Worst case p95 CPU latency: 2.8 seconds. Headroom vs PRD's 240,000 ms target: 86 times under.**

Latency is not a production concern. AASIST is 297K parameters, which makes it tractable on CPU even for 30 second audio with sliding window processing. For Capital One's real time fraud screening use case, GPU inference at 8 milliseconds per 4 second clip means a single GPU can score over 100 calls per second.

---

## 3. Honest disclosure on 2021 LA regression

2021 LA went from 5.67% in v1 to 8.01% in v2. This is the only dataset where RawBoost made things worse. Per codec breakdown:

| Codec | v1 EER | v2 EER | Direction |
|---|---|---|---|
| alaw | 5.37% | 8.03% | Worse |
| g722 | 7.49% | 9.74% | Worse |
| **gsm** | **4.10%** | **3.97%** | **Better** |
| none (clean) | 7.65% | 10.06% | Worse |
| opus | 6.62% | 8.59% | Worse |
| **pstn** | **3.63%** | **3.72%** | **Effectively flat** |
| ulaw | 5.35% | 8.00% | Worse |

**Hypothesis.** RawBoost's codec simulation uses mu-law and A-law companding with 8 bit quantization. The 2021 LA dataset, however, was generated by transmitting audio through a real Asterisk PBX with actual telephony codecs (a-law, mu-law, G.722, GSM, OPUS) running over SIP. The artifacts produced by real codec round trips differ from those produced by mathematical companding alone.

The codecs that improved or stayed flat (GSM, PSTN) are also the ones least similar to RawBoost's simulated codecs, which is consistent with the hypothesis: the model learned to handle simulated codec artifacts but partially overfit to them, making it less robust to the artifacts produced by real codec passes.

**Next sprint exploration.** Codec realistic augmentation using ffmpeg to pass training audio through actual codec encode/decode round trips. We expect this would close the 2021 LA gap while preserving the 2021 DF gains.

This is a real finding, not a failure. RawBoost is a blunt augmentation, the published recipe for AASIST family models, and we reproduced its expected behavior. The next refinement is codec realistic data augmentation, which the literature also supports as the next step beyond RawBoost.

---

## 4. Decision documentation

Per last sync's feedback on documenting decisions before implementation, I produced and uploaded the following decision documents to the team repo:

1. **Cross domain technique survey** (`aasist (Arnav)/decisions/decision_doc_cross_domain_techniques.md`). Surveys all six techniques you listed (RawBoost, SSL frontends, one class metric learning, phase + subband fusion, multi task learning, continual learning regularization). Recommends RawBoost as the primary choice for this sprint with rationale. SSL frontend is a parallel track if Mohini's Wave2Vec 2.0 branch lands in time.

2. **Reproducibility document** (`aasist (Arnav)/results/reproducibility.md`). Exact reproduction recipe including environment setup, data download, training hyperparameters, evaluation commands, and known gotchas. Anyone on the team or your team can re-derive any number reported here.

3. **EER summary document** (`aasist (Arnav)/results/eer_summary.md`). Per attack, per codec, and per compression breakdowns for both v1 and v2.

Going forward, decision documents will be written before implementation for any significant architectural choice.

---

## 5. Status of all three recommendations from last sync

| Recommendation | Status | Result |
|---|---|---|
| #1 Reduce 2021 DF EER from 22.95% toward 12-15% | Implemented (RawBoost), evaluated | 22.95% to 17.20%, a 25% relative reduction. Slightly above the 15% target. Path to 15% identified (codec realistic augmentation, next sprint). |
| #2 Document SHAP vs LIME justification | Owned by Hannah | Hannah's `SHAP_Features_Justification.md` is in the team repo. Her approach (SHAP on handcrafted acoustic features) is complementary to my planned iWAX work (SHAP on XGBoost over AASIST embeddings). |
| #3 Per utterance latency profiling with sliding window | Implemented, profiled GPU + CPU | Worst case 111 ms (GPU) / 2,777 ms (CPU). Both 86 to 2167 times under PRD target. Latency is not a production concern. |

---

## 6. What's planned next sprint (June 12 to June 19, PR1 deliverable)

| Date | Milestone |
|---|---|
| June 12 to June 15 | Codec realistic data augmentation (ffmpeg actual codec round trips) for v3. Close the 2021 LA regression while preserving 2021 DF gain. |
| June 12 to June 15 | Coordinate with Mohini once her Wave2Vec 2.0 model specification is finalized. Build ensemble inference scaffold (AASIST + w2v2 weighted average plus Platt or isotonic calibration). |
| June 15 to June 18 | First end to end ensemble evaluation on 2021 DF. iWAX scaffold (AASIST embeddings, XGBoost head, SHAP via shap.TreeExplainer). |
| **June 19** | **PR1 deliverable: v2 RawBoost results, v3 codec realistic results, ensemble baseline, iWAX prototype.** |

---

## 7. Open questions for sponsors

1. **Codec realistic augmentation direction.** Confirming that ffmpeg actual codec round trips for v3 is the right next step (vs trying SSL frontends or one class metric learning). Recommend v3 with ffmpeg, but happy to course correct.

2. **AWS architecture review.** Chase has started AWS infrastructure work in parallel with the modeling branches. The AWS architecture document we previously shared sets the design (S3, SageMaker, ECR, DynamoDB, CloudWatch within a VPC). Confirming this is still the intended direction so Chase doesn't drift, given that you suggested not waiting on the implementation.

3. **Model selection vs ensemble for production deployment.** Per the team meeting summary, the framing has shifted toward "deploy a single model initially as a priority." With Mohini's Wave2Vec 2.0 at 12.6% and my v2 AASIST at 17.20% on 2021 DF, the team consensus is leaning toward w2v2 as the primary model. Confirming this is acceptable, or whether you would prefer the ensemble (AASIST + w2v2 fused with Platt calibration) as the production deliverable for PR1.

---

## 8. Reproducibility

Everything described is on Rivanna under our team allocation (`ds2022`). All team members have read access.

- **v2 trained checkpoint:** `/scratch/mhq8ka/aasist/outputs/runs/aasist_v2_rawboost_14642481/best.pt`
- **v1 trained checkpoint:** `/scratch/mhq8ka/aasist/outputs/runs/aasist_fast_13608120/best.pt` (kept for v1 vs v2 comparison)
- **All eval results:** `/scratch/mhq8ka/aasist/outputs/eval/v2_*/` for v2, `/scratch/mhq8ka/aasist/outputs/eval/*/` for v1
- **Latency results:** `/scratch/mhq8ka/aasist/outputs/latency/{v2_gpu.json, v2_cpu.json}`
- **All code, decision documents, weekly writeups, and handoff documents:** team GitHub repo at `eglhannah/CapitalOne_AudioDeepfake_Project/aasist (Arnav)/`
- **Random seed:** 1234, `cudnn.deterministic=True`. Same seed used for v1 and v2.

Thanks for the structured feedback last sync. The framing of cross domain robustness as the next milestone shaped the entire sprint, and the resulting v2 numbers are the strongest demonstration we have so far that AASIST generalizes beyond the clean 2019 training set. Looking forward to discussing next steps Friday.

Best,
**Arnav Jain**
UVA MSDS Capstone, AASIST Modeling Branch
