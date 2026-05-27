# ASVspoof Dataset Overview: 2019 LA, 2021 LA Eval, and 2021 DF Eval

A comprehensive technical reference for UVA MSDS Capstone working on voice biometric fraud detection and spoofing countermeasures.

---

## Background and Challenge Context

The **ASVspoof** (Automatic Speaker Verification Spoofing and Countermeasures) initiative is a community-led benchmarking series that has run biannually since 2015. Its mission is to provide standardized datasets and evaluation platforms for the development of countermeasures (CMs) that protect automatic speaker verification (ASV) systems from spoofing attacks. The 2015 edition covered TTS and voice conversion (VC) attacks; 2017 focused on replay attacks; **2019 was the first edition to address all three major attack types — TTS, VC, and replay — in a single challenge.** The 2021 edition then pushed further toward real-world conditions by introducing channel variability and a new deepfake detection task.

The datasets described here — 2019 LA, 2021 LA eval, and 2021 DF eval — are the three most widely used benchmarks in the field and form the core training/evaluation regime for the vast majority of published spoofing countermeasure systems.

---

## Part 1: ASVspoof 2019 LA (Logical Access) Used for Model Training

### Overview

The 2019 LA subset is the cornerstone benchmark for TTS and VC spoofing detection. "Logical access" refers to scenarios where spoofed speech is injected **directly into the ASV system's input pipeline** (as opposed to physically replayed through a loudspeaker). Attacks are generated programmatically using state-of-the-art synthesis and conversion algorithms, and audio quality is uniformly high and clean — there is no channel noise, codec degradation, or reverberation, making it an ideal controlled laboratory benchmark.

### Source Corpus: VCTK

All bona fide speech in the 2019 database originates from the **CSTR VCTK corpus**, a collection of English read speech recorded in a hemi-anechoic chamber at the University of Edinburgh. The total speaker pool drawn upon for ASVspoof 2019 is **107 speakers: 46 male and 61 female**. Crucially, the data used to *train* the TTS and VC spoofing systems also comes from VCTK, but from a **disjoint set of speakers** — there is no overlap between the speakers in those training sets and the speakers whose voices appear in the ASVspoof 2019 corpus itself.

### Partition Structure

The LA dataset is split into three non-overlapping partitions. Speaker identity is entirely disjoint across partitions — a speaker who appears in training will not appear in development or evaluation.

| Partition | Speakers (M/F) | Bona fide utterances | Spoof utterances | Total utterances | Attacks present |
|-----------|----------------|----------------------|------------------|------------------|-----------------|
| Training  | 20 (8M, 12F)   | 2,580                | 22,800           | 25,380           | A01–A06 (known) |
| Development | 20 (8M, 12F) | 2,548                | 22,296           | 24,844           | A01–A06 (known) |
| Evaluation | 48 (21M, 27F) | 7,355                | 63,882           | 71,237           | A07–A19 (2 known + 11 unknown) |

**Notes on class imbalance:** Spoof utterances substantially outnumber bona fide utterances in every split (~9:1 ratio in training/development, ~8.7:1 in evaluation). This is an important practical consideration when designing loss functions and evaluation protocols. Average utterance duration is approximately 2–3 seconds.

The development split uses the same 20 speakers and attack systems as training — this enables threshold optimization and hyperparameter tuning. The evaluation set is meaningfully harder: it uses a much larger speaker pool (48 speakers) and introduces 11 **unknown attack systems** that the model has never seen during training.

### Attack Systems: Known vs. Unknown

A central design principle of ASVspoof 2019 is the known/unknown split. The goal is to benchmark generalization — a system that memorizes artifacts of A01–A06 but fails on novel attack algorithms is not considered robust.

**Known attacks (A01–A06)** — present in training and development, and partially in evaluation (A16 reuses the same algorithm as A04; A19 reuses the same algorithm as A06):

| Attack ID | Type | Method | Key Technology |
|-----------|------|---------|----------------|
| A01 | VC | Neural-network-based | Neural waveform conversion |
| A02 | VC | Spectral-filtering-based | Transfer function / spectral mapping |
| A03 | TTS | Waveform concatenation | Unit selection synthesis (MaryTTS) |
| A04 | TTS | Neural parametric (source-filter vocoder) | MERLIN + source-filter vocoder |
| A05 | TTS | Neural parametric (source-filter vocoder) | CURRENT toolkit |
| A06 | TTS | Neural parametric (WaveNet vocoder) | WaveNet waveform synthesis |

**Unknown attacks (A07–A19, excluding A16/A19 which are known references)** — appear only in the evaluation set. These were designed to stress-test generalization:

| Attack ID | Type | Key Technology / Description |
|-----------|------|------------------------------|
| A07 | TTS | Waveform concatenation synthesis |
| A08 | TTS | Neural TTS with WaveNet vocoder (pipeline) |
| A09 | TTS | Griffin-Lim waveform generation |
| A10 | TTS | End-to-end TTS with WaveRNN + ASV-transfer speaker encoder (among the hardest attacks — designed with ASV knowledge) |
| A11 | TTS | GAN-based waveform generation |
| A12 | VC | Neural waveform VC (pipeline, WaveNet-based) |
| A13 | VC | Moment-matching network VC + waveform filtering |
| A14 | TTS | Classical vocoder-based neural TTS |
| A15 | VC | Neural VC with WaveNet vocoder (pipeline) |
| A16 | TTS | Same waveform-concatenation algorithm as A04 (known reference) |
| A17 | VC | VAE-based VC with waveform filtering (hardest to detect by CM despite being weakest ASV threat) |
| A18 | VC | i-vector/PLDA VC with DNN glottal vocoder |
| A19 | VC | Same algorithm as A06 (known reference) |

The diversity of waveform generation techniques — classical vocoders, Griffin-Lim, GANs, WaveNet, WaveRNN, spectral filtering, and OLA (overlap-add) — makes A07–A19 a genuinely challenging benchmark for generalization. Some attacks (A10, A13) were found to severely degrade ASV reliability (EERs above 40–60%) while also being difficult to detect. A17, conversely, poses little threat to ASV but is the hardest to detect by countermeasures due to its subtle artifact profile.

### Bona Fide Data Creation

Bona fide utterances are drawn directly from VCTK recordings. Each speaker read approximately 400 sentences of English text (drawn from newspapers and phonetically rich sentence sets). Recordings were made in a hemi-anechoic chamber with a calibrated AKG microphone, resulting in clean, low-noise, broadband-quality audio at 48 kHz (downsampled to 16 kHz for ASVspoof use). The hemi-anechoic environment ensures that the intrinsic acoustic characteristics of each speaker's voice are preserved without room reverberation effects.

### Spoofed Data Creation

For each TTS or VC attack system:

- **TTS systems** were trained on VCTK speaker data (speakers disjoint from the 2019 corpus) and then used to synthesize utterances in the voice of target speakers from the 2019 corpus. The synthesis pipeline varies by attack: acoustic model (e.g., HMM-based or neural DNN), vocoder (source-filter, WaveNet, GAN), and post-processing (spectral filtering, waveform concatenation, etc.).
- **VC systems** were applied to bona fide utterances from source speakers to convert them into the voice of target speakers in the corpus. Conversion methods ranged from spectral mapping (transfer functions) to neural network-based conversion models.
- **Hybrid TTS-VC systems** (present in the unknown attacks) combine both paradigms, e.g. synthesizing speech and then applying VC refinement.

All spoofed utterances are generated such that the content (linguistic message) matches corresponding bona fide utterances where possible, to isolate the biometric/acoustic spoofing dimension. The resulting spoofed speech spans a wide range of perceptual quality — some utterances are readily identifiable as synthetic even by human listeners, while others were found in human assessment studies to be **indistinguishable from bona fide speech**, underscoring the real-world severity of the threat.

### Audio Format

All utterances are distributed as **16 kHz, 16-bit PCM FLAC** files. Labels are provided through per-split protocol text files specifying speaker ID, utterance ID, attack type, and the bona fide/spoof label.

---

## Part 2: ASVspoof 2021 LA Evaluation Set

### Motivation and Design Philosophy

The 2021 LA evaluation set was designed to bridge the gap between controlled laboratory conditions and real-world telephony deployment. While the 2019 LA corpus contained clean, artifact-free audio, real-world ASV systems must operate through phone networks, VoIP platforms, and various codecs that introduce channel-dependent distortions. The 2021 edition tests whether CMs remain robust when both bona fide and spoofed speech have been transmitted across actual telephony infrastructure.

**No new training data was provided for ASVspoof 2021.** Participants were required to use the ASVspoof 2019 LA training and development partitions for all CM training, and use of the 2019 evaluation partition was strictly forbidden. This intentional train/eval domain mismatch — clean training data vs. channel-corrupted evaluation — directly tests robustness and generalization.

### Source Data

Speech data for the 2021 LA evaluation set is sourced from the **ASVspoof 2019 LA evaluation partition** (itself derived from VCTK), with a substantial number of **new, previously unexposed bona fide utterances** collected from the same 48 speakers. New spoofed utterances were generated using the same 13 attack algorithms (A07–A19) as the 2019 LA evaluation set. The total speaker pool remains the same: **48 speakers (21 male, 27 female)**.

### Partition Counts

The 2021 LA evaluation is divided into a **progress subset** (used for intermediate leaderboard scoring during the challenge) and the full **evaluation subset**:

| Subset | Bona fide | Spoof | Total | Speakers (F/M) |
|--------|-----------|-------|-------|----------------|
| Progress | 1,676 | 14,788 | 16,464 | 37F + 30M |
| Evaluation | 14,816 | 133,360 | 148,176 | 37F + 30M |

The evaluation set is approximately 2× the size of the 2019 LA evaluation set in terms of unique speech content, magnified further by replication across codec conditions.

### Codec and Transmission Conditions

Every utterance appears under one of **seven transmission/codec conditions** (C1–C7), with C1 being the clean reference:

| Condition | Codec | Sampling Rate | Network | Bitrate |
|-----------|-------|---------------|---------|---------|
| C1 | None (clean reference) | 16 kHz | None | 250 kbps |
| C2 | a-law | 8 kHz | VoIP (Asterisk PBX) | 64 kbps |
| C3 | Unknown + µ-law | 8 kHz | PSTN → VoIP | Unknown |
| C4 | G.722 | 16 kHz | VoIP (Asterisk PBX) | 64 kbps |
| C5 | µ-law | 8 kHz | VoIP (Asterisk PBX) | 64 kbps |
| C6 | GSM Full Rate 6.10 | 8 kHz | VoIP (Asterisk PBX) | 13 kbps |
| C7 | OPUS | 16 kHz | VoIP (Asterisk PBX) | VBR ~16 kbps |

**C1** is identical to the ASVspoof 2019 LA scenario (no codec or channel effects), serving as a baseline reference. Conditions **C2 and C4–C7** involve transmission through an Asterisk private branch exchange (PBX) system using real SIP endpoints. Transmissions were made either within a local area network or across intercontinental connections (France to Italy or Singapore). **C3** uniquely reflects PSTN transmission originating from a mobile smartphone, involving multiple unknown transcodings — representing the hardest and most realistic telephony condition.

The codec choices span the range of legacy narrowband telephony (a-law G.711, µ-law, GSM at 8 kHz) and modern wideband codecs (G.722 at 16 kHz, OPUS with variable bitrate). No codec metadata was provided in file headers — CMs had to be codec-agnostic. Distribution of speakers and attack types is balanced across all conditions.

The progress subset contains conditions C1–C4; the evaluation subset adds C5–C7 as unseen conditions.

### Attack Systems

The attack systems are identical to those in the ASVspoof 2019 LA evaluation partition: attacks **A07–A19** (13 systems total, encompassing TTS, VC, and hybrid TTS-VC). These are the "unknown" attack systems from 2019's perspective, though challenge participants in 2021 have had access to the 2019 evaluation partition metadata (the attack labels) for analysis purposes.

### Key Research Challenge

The core challenge is designing CMs that are **simultaneously robust to diverse codec-induced channel distortion and diverse spoofing algorithm artifacts**. Channel effects can mask or alter the spectral features that CMs typically rely upon, such as phase discontinuities in concatenative TTS, unnatural spectral smoothness in neural TTS, or periodic artifacts from vocoders. Post-challenge results demonstrated that CMs trained on clean 2019 data were generally robust to the known codec conditions (C1–C4), but performance degraded under the more severe PSTN and low-bitrate GSM conditions (C3, C6).

---

## Part 3: ASVspoof 2021 DF (DeepFake) Evaluation Set

### Motivation and Scope

The DF task was **introduced for the first time in ASVspoof 2021** and represents a conceptual departure from the LA/PA tasks. While LA and PA are designed around protecting a specific ASV system from attack, the DF task addresses a broader societal problem: **detecting fabricated speech posted online** (in news media, social platforms, television), regardless of whether an ASV system is involved. The adversary's goal is not to bypass identity verification but to fabricate convincing voice content in a target speaker's voice.

This makes the DF task a standalone CM problem — there is **no tandem ASV system** and the primary metric is simply the **Equal Error Rate (EER)**.

The scenario simulates the distribution chain of online media: synthetic or converted speech content is compressed and re-encoded using lossy codecs before being uploaded or broadcast, matching real-world content on social media and news platforms.

### Source Data and Speakers

The DF evaluation set draws from **multiple source datasets**, making it more heterogeneous than the LA tasks:

- **ASVspoof 2019 LA evaluation partition** (VCTK-derived, the primary source)
- **Voice Conversion Challenge (VCC) 2018** data — contributing 12 additional speakers
- **Voice Conversion Challenge (VCC) 2020** data — contributing 14 additional speakers
- Additional, undisclosed source data revealed after the challenge evaluation phase

Total speaker count for the DF evaluation set: **50 female and 43 male speakers** (93 total) — significantly more than the LA evaluation sets, driven by the inclusion of VCC 2018/2020 speakers.

The spoofed utterances include a **much larger variety of attack algorithms** than used in the LA tasks, including over 100 different TTS and VC systems — many not previously encountered in any ASVspoof challenge. This is the broadest attack diversity of the three datasets discussed here.

### Partition Counts

| Subset | Bona fide | Spoof | Total | Speakers (F/M) |
|--------|-----------|-------|-------|----------------|
| Progress | 5,768 | 53,557 | 59,325 | 37F + 30M |
| Evaluation | 14,869 | 519,059 | 533,928 | 50F + 43M |

The evaluation set is very large — over 533,000 utterances — driven by the Cartesian product of multiple spoofing algorithms × multiple speakers × multiple codec conditions. The severe class imbalance (~35:1 spoof-to-bonafide ratio) is more extreme than in 2019 LA.

### Codec Compression Conditions

Unlike the LA task, which simulates telephony transmission, the DF task simulates **media compression for storage and distribution** (MP3, M4A, OGG formats typical of social media and broadcast platforms):

| Condition | Compression | VBR Range |
|-----------|-------------|-----------|
| DF-C1 | None (clean reference) | — |
| DF-C2 | MP3 (lower VBR) | Low |
| DF-C3 | MP3 (higher VBR) | High |
| DF-C4 | M4A / AAC (lower VBR) | Low |
| DF-C5 | M4A / AAC (higher VBR) | High |
| DF-C6 | OGG Vorbis (lower VBR) | Low |
| DF-C7 | OGG Vorbis (higher VBR) | High |
| DF-C8 | Undisclosed (revealed post-challenge) | — |
| DF-C9 | Undisclosed (revealed post-challenge) | — |

Audio processing used `ffmpeg` and `sox`. Codec conditions C8 and C9 were withheld during the challenge and revealed afterward. No codec metadata was provided to participants — systems had to be blind to compression type and settings. Each condition also includes multiple vocoder sub-conditions reflecting the diversity of synthesis back-ends in the spoofed data.

### Attack Diversity

The DF evaluation set includes attacks generated by **over 100 distinct TTS and VC systems**, encompassing a broad range of model architectures:

- **Statistical models**: HMM-based TTS, GMM-UBM statistical parametric synthesis
- **Neural TTS**: Tacotron, FastSpeech, and related sequence-to-sequence models with various neural vocoders (WaveNet, WaveGlow, WaveRNN, HiFi-GAN, etc.)
- **Neural VC**: AutoVC, StarGAN-VC, VQVAE-based VC, CycleGAN-VC, and others
- **End-to-end models**: Models trained jointly for acoustic and waveform synthesis
- **VCC 2018 and VCC 2020 VC systems**: A large collection of community-submitted systems from those challenges, covering many architectures not seen in ASVspoof 2019

This diversity is the key challenge of the DF task: CMs must generalize not just across codec conditions but across a fundamentally larger and more heterogeneous space of spoofing algorithms than they have seen during training.

### Key Differences from 2021 LA

| Dimension | 2021 LA Eval | 2021 DF Eval |
|-----------|-------------|-------------|
| ASV system involved | Yes (tandem CM+ASV) | No (standalone CM) |
| Primary metric | min t-DCF | EER |
| Channel distortion type | Telephony (VoIP/PSTN) | Media compression (MP3/AAC/OGG) |
| Attack systems | A07–A19 (13 systems) | 100+ TTS/VC systems |
| Speaker sources | VCTK only | VCTK + VCC 2018 + VCC 2020 + other |
| Domain mismatch | Training: clean, eval: codec-corrupted | Training: clean+2019LA, eval: compressed multi-source |
| Scale | ~148K utterances | ~534K utterances |

---

## Cross-Dataset Summary Table

| Property | 2019 LA Train | 2019 LA Dev | 2019 LA Eval | 2021 LA Eval | 2021 DF Eval |
|----------|--------------|------------|-------------|-------------|-------------|
| Bona fide count | 2,580 | 2,548 | 7,355 | 14,816 | 14,869 |
| Spoof count | 22,800 | 22,296 | 63,882 | 133,360 | 519,059 |
| Total utterances | 25,380 | 24,844 | 71,237 | 148,176 | 533,928 |
| # Speakers | 20 | 20 | 48 | 48 | 93 |
| # Attack systems | 6 | 6 | 13 | 13 | 100+ |
| Channel distortion | None | None | None | Telephony codecs (7 conditions) | Media codecs (9 conditions) |
| Source corpus | VCTK | VCTK | VCTK | VCTK | VCTK + VCC2018 + VCC2020 |
| Spoof:bonafide ratio | ~8.8:1 | ~8.7:1 | ~8.7:1 | ~9.0:1 | ~34.9:1 |

---

## Citation References

For the 2019 LA dataset:
> Yamagishi, J. et al. (2019). *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech.* Computer Speech & Language, 64, 101114. (arXiv:1911.01601)

For the 2021 LA and DF evaluation sets:
> Liu, X. et al. (2023). *ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild.* IEEE/ACM Trans. Audio, Speech, and Language Processing. (arXiv:2210.02437)

Challenge overview paper:
> Yamagishi, J. et al. (2021). *ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection.* Proc. ASVspoof Workshop. (arXiv:2109.00537)
