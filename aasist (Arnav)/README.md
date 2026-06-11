# AASIST Modeling Branch — Arnav Jain

This folder contains the AASIST audio anti-spoofing modeling workstream for the UVA × Capital One Voice Anti-Spoofing capstone.

**Owner:** Arnav Jain (arnav.jain321@gmail.com)
**Workstream:** AASIST modeling branch + project coordination
**Reference architecture:** AASIST (Jung et al., ICASSP 2022; arXiv:2110.01200)
**Reference implementation:** [clovaai/aasist](https://github.com/clovaai/aasist)
**Status:** v1 complete; v2 (RawBoost) complete; v3 (codec realistic augmentation) next sprint

---

## Contents at a glance

| Folder | What's in it |
|---|---|
| `code/` | Training scripts (v1 + v2 with RawBoost), evaluation scripts for all 4 ASVspoof datasets, Slurm sbatch wrappers, plot generators |
| `decisions/` | Decision documents — cross-domain improvement technique survey (per Mustafa Rec #1) |
| `results/` | Degradation curve plot, training curve plot, EER summary, reproducibility notes |
| `weekly_updates/` | Weekly sponsor sync writeups (markdown source + PDF) |
| `handoff/` | Documentation for team members who need to load / use the trained checkpoint |

---

## Headline results

### v1 (baseline, no augmentation)

AASIST trained on ASVspoof 2019 LA (25 epochs, A6000, 3.3 hr wall), evaluated across four progressively harder test sets:

| Test set | v1 EER | What's hard about it |
|---|---|---|
| 2019 LA dev | **0.90%** | In-domain (seen attacks) |
| 2019 LA eval | **3.33%** | Unknown attacks (A07-A19) |
| 2021 LA eval | **5.67%** | + Telephony codecs (7 conditions) |
| 2021 DF eval | **22.95%** | + Media compression + 100+ unseen TTS/VC systems |

### v2 (RawBoost augmentation) — current canonical model

Same architecture as v1, retrained with RawBoost data augmentation per Mustafa Rec #1 (June 9 email):

| Test set | v1 EER | v2 EER | Direction |
|---|---|---|---|
| 2019 LA dev | 0.90% | **0.78%** | Better |
| 2019 LA eval | 3.33% | **1.87%** | Better (-44% relative) |
| 2021 LA eval | 5.67% | 8.01% | Worse (one regression, see Section 3 of writeup) |
| **2021 DF eval** | **22.95%** | **17.20%** | **Better (-25% relative)** |

All four under PRD's 25% gate for both v1 and v2. v2 improves on 3 of 4 datasets. The hardest published attacks (A17, A18) improved by 22 and 14 percentage points respectively.

See `results/aasist_v1_vs_v2_degradation_curve.png` for the visual, `results/eer_summary.md` (v1) and `results/eer_summary_v2.md` (v2) for per-attack and per-codec breakdowns, and `results/reproducibility.md` for exact configs.

---

## Where the checkpoint lives (not in git — too large for repo hygiene)

- **Trained checkpoint:** `/scratch/mhq8ka/aasist/outputs/runs/aasist_fast_13608120/best.pt` on Rivanna
- **Per-utterance predictions:** `/scratch/mhq8ka/aasist/outputs/eval/{2019_LA, 2021_LA, 2021_DF}_eval/predictions.csv` on Rivanna

Team members have read access (chmod-ed). Loading instructions in `handoff/AASIST_checkpoint_handoff_for_Hannah.md`.

---

## v2 RawBoost sprint (complete)

In response to sponsor recommendation to close the 2021 DF gap from 22.95% to under 15%, the v2 model was trained with RawBoost data augmentation. See `decisions/decision_doc_cross_domain_techniques.md` for the survey of six techniques considered and the rationale for choosing RawBoost.

Result: 2021 DF dropped from 22.95% to 17.20% (-25% relative). Slightly above the 15% target. Path to 15% identified for v3 (codec realistic augmentation via ffmpeg actual codec round trips).

| File | Purpose |
|---|---|
| `code/rawboost.py` | Augmentation module: 4 transforms (linear convolutive noise, impulsive noise, stationary noise, codec simulation via mu-law/A-law) |
| `code/train_aasist_v2.py` | V2 training script with RawBoost integrated as a thin wrapper over Chase's `ASVspoofLADataset` |
| `code/sbatch_aasist_v2_rawboost.sh` | Slurm submission for v2 training |
| `code/run_eval_v2_*.sh` | v2 evaluation wrappers for 2019 LA dev/eval and 2021 LA |
| `code/sbatch_eval_v2_*.sh` | Slurm submissions for v2 evaluation |
| `code/latency_profile.py` | Sliding window latency profiler (Mustafa Rec #3) |
| `code/sbatch_latency_profile.sh` | Slurm submission for latency profiling on GPU + CPU |

Training wall time: ~3 hr on NVIDIA A6000. Run name: `aasist_v2_rawboost_14642481`.

## Latency profiling result

- **GPU (A6000), worst case p95:** 111 ms on 30 second utterance (14 sliding windows). 2167 times under PRD's 4 minute target.
- **CPU, worst case p95:** 2.8 seconds on 30 second utterance. 86 times under target.

Latency is not a production concern. See `results/eer_summary_v2.md` Section 5 for the full table.

---

## Change log

| Date | Change | Author |
|---|---|---|
| 2026-06-11 | Pushed v2 artifacts: RawBoost code, all 4 v2 eval results, latency profiling on GPU + CPU, v1 vs v2 plot, June 12 sponsor writeup. Updated sbatch scripts to use new ds2022 allocation. | Arnav |
| 2026-06-10 | Completed v2 evaluation on all 4 datasets. 2021 DF: 22.95% to 17.20%. A17: 31.82% to 10.13%. A18: 40.93% to 27.38%. | Arnav |
| 2026-06-09 | Trained v2 AASIST with RawBoost (25 epochs, A6000, 3 hr). Dev EER 0.78%. Initial upload of v1 workstream to team repo per sponsor feedback. | Arnav |
| 2026-06-05 | Sent sponsor writeup PDF to Mustafa, Mehul, Arindam in lieu of attending sync | Arnav |
| 2026-06-01 | Completed 2021 DF cross-domain eval (22.95% EER, 100% file coverage after ffmpeg fallback added) | Arnav |
| 2026-05-31 | Completed 2019 LA eval (3.33%) and 2021 LA eval (5.67%) | Arnav |
| 2026-05-29 | Trained v1 AASIST on 2019 LA, achieved 0.90% dev EER | Arnav |

---

## Coordination notes

- **Chase:** Code reuses his `asv_baseline` modules (protocol parser, audio loader, EER metrics) without modification. His pipeline is the foundation.
- **Hannah:** Has read access to the trained checkpoint. Per-utterance prediction CSVs are her ready-made input for fairness slicing and FR-12 deliverable.
- **Mohini:** Wave2Vec 2.0 branch is the second PRD-mandated branch. Ensemble integration planned once her trained model + spec are confirmed.

---

## Quick reference for the team

Want to use AASIST yourself? Start with `handoff/AASIST_checkpoint_handoff_for_Hannah.md` — it has copy-pasteable code to load the model and score audio. Despite the name, it works for any team member.
