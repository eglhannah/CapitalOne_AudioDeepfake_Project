# AASIST Modeling Branch — Arnav Jain

This folder contains the AASIST audio anti-spoofing modeling workstream for the UVA × Capital One Voice Anti-Spoofing capstone.

**Owner:** Arnav Jain (arnav.jain321@gmail.com)
**Workstream:** AASIST modeling branch + project coordination
**Reference architecture:** AASIST (Jung et al., ICASSP 2022; arXiv:2110.01200)
**Reference implementation:** [clovaai/aasist](https://github.com/clovaai/aasist)
**Status:** v1 complete; v2 (RawBoost) in progress

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

## Headline result

AASIST trained on ASVspoof 2019 LA (25 epochs, A6000, 3.3 hr wall), evaluated across four progressively harder test sets:

| Test set | EER | What's hard about it |
|---|---|---|
| 2019 LA dev | **0.90%** | In-domain (seen attacks) |
| 2019 LA eval | **3.33%** | Unknown attacks (A07–A19) |
| 2021 LA eval | **5.67%** | + Telephony codecs (7 conditions) |
| 2021 DF eval | **22.95%** | + Media compression + 100+ unseen TTS/VC systems |

All four under the PRD's 25% EER gate. Monotonic graceful degradation.

See `results/degradation_curve.png` for the visual, `results/eer_summary.md` for per-attack and per-codec breakdowns, and `results/reproducibility.md` for exact configs.

---

## Where the checkpoint lives (not in git — too large for repo hygiene)

- **Trained checkpoint:** `/scratch/mhq8ka/aasist/outputs/runs/aasist_fast_13608120/best.pt` on Rivanna
- **Per-utterance predictions:** `/scratch/mhq8ka/aasist/outputs/eval/{2019_LA, 2021_LA, 2021_DF}_eval/predictions.csv` on Rivanna

Team members have read access (chmod-ed). Loading instructions in `handoff/AASIST_checkpoint_handoff_for_Hannah.md`.

---

## Current sprint (v2 RawBoost)

In response to sponsor recommendation to close the 2021 DF gap from 22.95% to ≤15%, the v2 model is being trained with RawBoost data augmentation. See `decisions/decision_doc_cross_domain_techniques.md` for the survey of six techniques considered and the rationale for choosing RawBoost.

| File | Purpose |
|---|---|
| `code/rawboost.py` | Augmentation module — 4 transforms: linear convolutive noise, impulsive noise, stationary noise, codec simulation (μ-law/A-law) |
| `code/train_aasist_v2.py` | V2 training script with RawBoost integrated as a thin wrapper over Chase's `ASVspoofLADataset` |
| `code/sbatch_aasist_v2_rawboost.sh` | Slurm submission for v2 training |

Expected wall time: ~3–5 hr on a single GPU. Run name pattern: `aasist_v2_rawboost_<jobid>`.

---

## Change log

| Date | Change | Author |
|---|---|---|
| 2026-06-09 | Initial upload of full workstream to team repo per sponsor feedback (decision docs, code, results, weekly updates) | Arnav |
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
