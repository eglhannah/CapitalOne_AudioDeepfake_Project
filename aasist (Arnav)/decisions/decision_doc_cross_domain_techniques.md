# Decision Document — Cross-Domain Improvement Techniques for AASIST

**Workstream:** AASIST Modeling Branch — closing the 2021 DF cross-domain gap
**Author:** Arnav Jain
**Date:** 2026-06-09
**Status:** Recommendation locked — adoption pending implementation start

**Change log:**
- 2026-06-09 (Arnav): Initial decision document in response to sponsor recommendation #1 (Mustufa Zaranwala, email 2026-06-09).

---

## Context

Current AASIST results (best dev EER 0.90% on 2019 LA, 22.95% on 2021 DF eval — below the PRD's 25% gate but with significant headroom for improvement). Mustufa correctly noted the 22.95% is expected given training on clean 2019 LA, and asked us to survey techniques to close the cross-domain gap to a more acceptable level (12–15% or below).

This document surveys six techniques he listed, scores each on five dimensions, and recommends one for adoption. The selected technique will be implemented in the current sprint and re-evaluated across all four datasets (2019 LA dev, 2019 LA eval, 2021 LA, 2021 DF) in time for PR1 (June 19).

---

## 1. Decision matrix

| Decision driver | Candidates considered | Actual decision | Implications & tradeoffs |
|---|---|---|---|
| Close cross-domain gap on 2021 DF (22.95% → target ≤15%) | RawBoost, SSL Frontends, One-Class Metric Learning, Phase + Subband Fusion, Multi-Task Learning, Continual Learning Regularization | **RawBoost (primary)** with **SSL Frontend exploration as a parallel track if Mohini's w2v2 lands in time** | RawBoost: highest published lift on 2021 DF for AASIST (~50% reduction in EER), lowest implementation effort (1–2 days), no architecture change, drops into existing dataloader. Tradeoff: needs hyperparameter tuning for our setup; doesn't address fundamental representation limitations. Backup: SSL Frontend gives larger ceiling but needs Mohini's branch + more implementation effort. |

---

## 2. Technique-by-technique survey

### 2.1 RawBoost

**What it does.** RawBoost is a training-time data augmentation pipeline for raw-waveform anti-spoofing models. It applies four families of waveform-level transformations during training to simulate the kinds of channel and compression effects that appear in real-world deployment but not in clean ASVspoof 2019 LA training data.

**Specific transformations.**
- **Linear convolutive noise:** randomly chosen FIR filters simulating channel responses
- **Impulsive signal-dependent additive noise:** sparse impulse noise overlaid on the audio
- **Stationary signal-independent additive noise:** broadband colored noise
- **Codec / coding-decoding simulation:** simulated narrowband telephony effects (μ-law, GSM-like)

Applied randomly to each training utterance (with configurable probabilities), creating a much larger and more diverse effective training distribution.

**Mechanism for closing the gap.** AASIST trained on 2019 LA learns to detect spoof artifacts in a clean spectro-temporal regime. When inference hits codec-corrupted or compressed audio (2021 LA, 2021 DF), the artifacts the model relies on are partially destroyed, and the model degrades. RawBoost teaches the model to detect spoof artifacts that survive realistic channel transformations — exactly the conditions the 2021 sets simulate.

**Published lift on 2021 DF.** AASIST + RawBoost is the published baseline pairing that achieves ~12–15% EER on 2021 DF (vs ~22–25% without). Lift is roughly **50% reduction in EER** on the same model architecture. This is the largest single-technique lift in the published 2021 anti-spoofing literature for the AASIST family.

**Implementation effort.** Low (1–2 days). RawBoost is implemented as a `torch.utils.data.Dataset` wrapper that applies augmentation during the `__getitem__` call. The reference implementation is open-source (`TakHemlata/RawBoost`). Integrates cleanly with our existing `ASVspoofLADataset` and `train_aasist.py`.

**Risks / tradeoffs.**
- Requires tuning augmentation probabilities (training instability if augmentation is too aggressive)
- Slightly slower training step (additional preprocessing during dataloader)
- Does not address fundamental representation limits — model still learns the same architecture, just on a richer distribution

**Citation.** Tak et al. (2022). *RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing.* ICASSP 2022. [arXiv:2111.04433]

---

### 2.2 SSL Frontends (Self-Supervised Learning)

**What it does.** Replaces AASIST's learned sinc-conv front-end with a pretrained self-supervised speech representation model (Wave2Vec 2.0, HuBERT, WavLM, or XLS-R). The pretrained encoder's frozen (or fine-tuned) representations are fed into the AASIST graph attention layers instead of raw-waveform-derived features.

**Mechanism for closing the gap.** SSL encoders are pretrained on tens of thousands of hours of unlabeled speech across many domains (LibriSpeech, VoxPopuli, Common Voice, multilingual VoxLingua), so their representations are already robust to channel and recording variations. Anti-spoofing trained on top of these representations inherits this robustness for free.

**Published lift on 2021 DF.** XLS-R + AASIST achieves ~5–8% EER on 2021 DF in published work — a **larger lift than RawBoost alone**. Wave2Vec 2.0 + AASIST gets ~8–12%. The result holds across multiple papers and is the current SOTA family for cross-domain anti-spoofing.

**Implementation effort.** Medium-high (3–5 days). Requires:
- Download pretrained checkpoint (XLS-R is ~1–2 GB, multilingual)
- Modify AASIST input layer to accept SSL encoder output instead of raw waveform
- Decide freeze vs fine-tune for encoder (literature recommends fine-tune all layers with low LR)
- Significantly larger model (300M+ params total) — requires longer training, more GPU memory
- Need to retrain from scratch with the new front-end

**Risks / tradeoffs.**
- Substantially larger model → longer training (~12–24 hr) and higher memory
- Significant architectural change (AASIST head needs to be re-validated with new input dims)
- **Strong overlap with Mohini's Wave2Vec 2.0 branch** — if pursued, makes sense to coordinate with her so we don't duplicate work

**Citation.** Tak, H., et al. (2022). *Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2Vec 2.0 and Data Augmentation.* Odyssey 2022.

---

### 2.3 One-Class Metric Learning

**What it does.** Replaces the standard binary classification loss with a one-class objective (OCSoftmax, OC-AASIST). The model learns a tight bonafide manifold in embedding space; spoof samples are pushed outside a margin. Detection is done by distance to the bonafide centroid rather than binary score.

**Mechanism for closing the gap.** Binary classification overfits to specific spoof artifacts seen during training. One-class learning instead focuses on what bonafide audio looks like — which is more stable across domains. Unseen attacks (2021 DF has 100+) become "outliers from the bonafide manifold" rather than "things the model has seen as spoofs," which is closer to how the model should reason about novel attacks.

**Published lift on 2021 DF.** ~3–4 percentage point improvement over binary baseline on 2021 LA (lift on 2021 DF less consistently documented). OC-Softmax is widely cited but not the top-published technique for AASIST on 2021 DF specifically.

**Implementation effort.** Medium (2–3 days). Loss function rewrite, hyperparameter re-tuning, threshold-selection on dev set.

**Risks / tradeoffs.**
- Requires careful hyperparameter tuning (margin, embedding dim, weight on each loss component)
- Score interpretation changes from probability to distance — calibration approach differs
- Less compatible with downstream score-fusion for ensemble (we'd need to re-derive a probability scale)

**Citation.** Zhang et al. (2021). *One-class learning towards synthetic voice spoofing detection.* IEEE SPL.

---

### 2.4 Explicit Phase and Subband Feature Fusion

**What it does.** Adds explicit signal-processing features alongside the learned waveform features. AASIST learns features from raw waveform, but some spoof artifacts live in the phase spectrum or specific frequency subbands that learned features may not optimally capture. Phase fusion concatenates handcrafted phase-based features (e.g., constant-Q phase, group delay) with AASIST embeddings; subband fusion does the same with frequency-subband energy features.

**Mechanism for closing the gap.** Generative TTS/VC systems often have characteristic phase artifacts (unnatural phase coherence, missing fine-grained phase structure). These are not always well-captured by AASIST's raw-waveform learning. Adding explicit phase features gives the classifier a more complete view.

**Published lift on 2021 DF.** Mixed; lift of 1–3 percentage points reported in some papers, no consistent SOTA. More incremental than transformative.

**Implementation effort.** Medium-high (3–5 days). Requires implementing handcrafted feature extraction (constant-Q transform, group delay, mel-subband energies), fusion strategy (early concat vs late concat vs attention fusion), retraining with combined feature input.

**Risks / tradeoffs.**
- Substantial architecture change (input dim grows; new fusion layer)
- Pushing handcrafted features into a learned model goes against the end-to-end raw-waveform philosophy of AASIST
- Modest empirical lift compared to RawBoost or SSL

**Citation.** Various — see Liu et al. (2022) and references in ASVspoof 2021 challenge analyses.

---

### 2.5 Multi-Task Learning (Utterance vs Segmental Level)

**What it does.** Trains AASIST simultaneously on two related objectives:
1. **Utterance-level:** the standard "is this whole clip bonafide or spoof?" binary task
2. **Segmental-level:** "for each 100ms (or similar) segment within the clip, is this segment bonafide or spoof?"

The two tasks share representations, with a shared backbone and two heads.

**Mechanism for closing the gap.** Forces the model to localize spoof artifacts temporally rather than relying only on aggregate utterance-level features. Acts as regularization, encouraging more robust representations.

**Published lift on 2021 DF.** ~1–2 percentage points in published work. Useful as a complement to other techniques but not a standalone winner.

**Implementation effort.** Medium-high (2–4 days). Requires segment-level labels (we'd need to derive them from utterance-level labels — typically "all frames in a spoof utterance are labeled spoof" which is a weak label).

**Risks / tradeoffs.**
- Weak segment labels limit the technique's effectiveness in our setup (we don't have true segment-level annotations)
- Significantly more complex training loop
- Modest expected lift

**Citation.** Various MTL extensions in the ASVspoof 2021 literature.

---

### 2.6 Continual Learning Regularization

**What it does.** Uses regularization techniques from continual learning (Elastic Weight Consolidation, Learning without Forgetting, replay buffers) to allow the model to adapt to new data without catastrophic forgetting of older data. Primarily useful when fine-tuning a pretrained AASIST on additional data progressively.

**Mechanism for closing the gap.** Limited applicability to our setup. CL is most useful when training data arrives in a sequence (e.g., new attack types appear over time and we want to fine-tune without forgetting old ones). Our setup is a single-shot training on a fixed 2019 LA dataset, so CL doesn't directly address the cross-domain gap.

**Published lift on 2021 DF.** Niche; minimal direct evidence on cross-domain DF generalization. CL is more relevant for production deployment over time, not for closing the train-test domain gap.

**Implementation effort.** Medium (2–4 days), but unclear payoff for our use case.

**Risks / tradeoffs.**
- Solves a problem we don't have (sequential data arrival)
- May still help marginally as a regularization technique
- Lower priority than the other five techniques

**Citation.** Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS (EWC reference).

---

## 3. Comparative summary

| Technique | Published lift on 2021 DF | Impl. effort | Architecture change | Best evidence for AASIST family |
|---|---|---|---|---|
| **RawBoost** | **High (~50% EER reduction)** | **Low (1–2 d)** | **None** | **Direct, strong** |
| SSL Frontends | Highest (50–70% reduction) | Medium-high (3–5 d) | Major (new encoder) | Direct, strong |
| One-Class Metric Learning | Moderate (3–4 pp) | Medium (2–3 d) | Loss + threshold | Indirect; no clear SOTA for AASIST |
| Phase + Subband Fusion | Low-moderate (1–3 pp) | Medium-high (3–5 d) | Substantial (new input path) | Mixed |
| Multi-Task Learning | Low (1–2 pp) | Medium-high (2–4 d) | Moderate (new head, weak labels) | Modest |
| Continual Learning | Niche / unclear | Medium (2–4 d) | Loss regularization | Limited applicability |

---

## 4. Recommendation

### Primary: RawBoost

For the current sprint (deadline PR1 = June 19), **RawBoost is the right choice**:

- **Highest impact-per-day-of-engineering** among the six candidates
- Reference implementation is open-source (`TakHemlata/RawBoost`)
- Drops into our existing `ASVspoofLADataset` with no architecture change
- Strong published evidence: AASIST + RawBoost is the canonical published improvement path on 2021 DF
- Allows us to deliver a measurable cross-domain improvement (22.95% → expected 12–15%) in time for PR1
- Mustufa listed it first in his recommendations — aligns with sponsor intuition

### Secondary / parallel track: SSL Frontend exploration

The SSL frontend approach has a higher ceiling (~5–8% on DF) but requires more implementation effort and overlaps with **Mohini's Wave2Vec 2.0 workstream**. Rather than duplicate her work, the right play is:

- **If Mohini's w2v2 branch lands with a trainable model in time (~before PR1)**, evaluate an SSL-frontend variant of AASIST (XLS-R or w2v2 as the encoder, AASIST graph attention head) as a stretch goal.
- **If timing doesn't work**, defer SSL frontend exploration to post-PR1 and rely on RawBoost as the primary cross-domain technique.

This avoids duplicate work on the team and keeps Mohini's branch on its own timeline.

### Rejected / deferred

- **One-Class Metric Learning** — moderate effort, moderate lift, complicates ensemble integration. Defer.
- **Phase + Subband Fusion** — high effort, low-moderate lift. Defer.
- **Multi-Task Learning** — limited by lack of true segment labels. Defer.
- **Continual Learning** — solves a problem we don't have. Defer to post-capstone if relevant for deployment.

---

## 5. Implementation plan for RawBoost

| Week | Task |
|---|---|
| Jun 9 – Jun 11 | Fork `TakHemlata/RawBoost` reference implementation. Wire `RawBoostAugment` into the `ASVspoofLADataset` as an optional transform with configurable probabilities. Unit test on a single clip. |
| Jun 11 – Jun 14 | Retrain AASIST + RawBoost from scratch on 2019 LA train (25 epochs, same hyperparameters as v1 plus augmentation enabled with probabilities 0.5 / 0.5 / 0.5 / 0.5 for the four families). Track dev EER each epoch. |
| Jun 14 – Jun 16 | Re-evaluate the retrained model on all four datasets (2019 LA dev, 2019 LA eval, 2021 LA, 2021 DF). Compare v1 vs v2 EER side by side. |
| Jun 16 – Jun 18 | Update plots and writeup; produce a v2 degradation curve; document the lift in the decision-doc framework. |
| **Jun 19 – PR1** | Both v1 and v2 results presented; degradation curve shows progress. |

---

## 6. Success criteria

- **2021 DF EER drops from 22.95% to ≤15%** (Mustufa's target). Quantitatively measurable.
- **2019 LA dev EER does not regress below 1%** (we don't want to trade in-domain accuracy for cross-domain robustness).
- **Per-codec breakdown on 2021 LA remains tight** (the robustness story holds).
- **Per-attack pattern preserves the published A09-trivial, A17/A18-hardest signature.**

If RawBoost achieves all four criteria, it's adopted as the final cross-domain technique for the capstone deliverable. If it falls short of the 15% DF target, we re-open this decision document and consider stacking SSL frontend (subject to Mohini's branch timing).

---

## References

- Tak, H., Kamble, M., Patino, J., Todisco, M., & Evans, N. (2022). *RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing.* ICASSP 2022. [arXiv:2111.04433]
- Tak, H., Todisco, M., Wang, X., Jung, J.-w., Yamagishi, J., & Evans, N. (2022). *Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2Vec 2.0 and Data Augmentation.* Odyssey 2022.
- Zhang, Y., Jiang, F., & Duan, Z. (2021). *One-class learning towards synthetic voice spoofing detection.* IEEE Signal Processing Letters.
- ASVspoof 2021 challenge analyses and survey papers (Liu et al., 2023).
- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS (Continual Learning).
