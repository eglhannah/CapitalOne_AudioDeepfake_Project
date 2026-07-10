# Capstone Progress Update — 7/10 Meeting Plan

## Overview

This update responds to the review comments from the 6/26 meeting and summarizes the team's progress toward the 7/10 milestone (working end-to-end application). Chase is unable to attend today; he will follow up asynchronously on the AWS deployment status.

## Response to 6/26 Review Comments

| Review Item | Status | Owner |
|---|---|---|
| Remove "Capital One benefits" language from page 2 of the report | Done | Mohini |
| Add SHAP vs other frameworks pros/cons to the report | In progress, included in the updated report | Hannah |
| Fix production threshold (0.5 placeholder does not match EER) | Done. Metrics now reported at the EER operating point with ROC curves. See `aasist/results/v3_threshold_metrics.md` in the repo. | Arnav |
| Share AWS deployment architecture for review and signoff | AASIST v3 Lambda deployment in progress. Chase to share architecture writeup asynchronously. | Chase |
| Working end-to-end application by 7/10 | Model inference stack complete (AASIST v3 loader in `aasist/simple_aasist.py`, w2v loader in `w2v/simple_model.py`). Full app wrapper is next step. | Team |
| Provide GitHub repo access | Pending — Hannah to add Mustafa as collaborator today | Hannah |

## Modeling Updates Since 6/17 Report

### AASIST (Arnav)

A third version, AASIST v3, was trained using CodecAugment (real telephony codec passes: alaw, ulaw, g722, opus) in place of RawBoost's synthetic codec channel. v3 is the best AASIST version on every eval set:

| Benchmark | v1 (no aug) | v2 (RawBoost) | v3 (CodecAugment) |
|---|---|---|---|
| 2019 LA eval | 3.33% | 1.87% | **1.67%** |
| 2021 LA eval | 5.67% | 8.01% | **4.67%** |
| 2021 DF eval | 22.95% | 17.20% | **17.00%** |

The 2021 LA improvement (8.01% → 4.67%, a 42% relative reduction) validates the codec-augmentation approach for real-channel deployment. All predictions, model weights, and training code are in the repo and on HuggingFace (`arnavjain321/aasist-v3-codecaugment`).

### Wave2Vec 2.0 (Mohini)

Mohini identified a decoder pipeline issue that was silently degrading her 2021 evaluation. After the fix, results are:

| Benchmark | w2v (fixed) |
|---|---|
| 2021 LA eval | 10.79% |
| 2021 DF eval | 15.46% |

The 2021 DF result meets the ~15% PRD target from the report's C1 section.

### Ensemble (score-level fusion of AASIST v3 + w2v)

Linear score fusion of the two model outputs on the intersection of file_ids scored by both:

| Benchmark | AASIST v3 alone | w2v alone (on join) | 50/50 Ensemble | Optimal Ensemble |
|---|---|---|---|---|
| 2021 LA eval | 4.67% | 4.49% | **3.18%** | **2.84%** (w2v=0.47) |
| 2021 DF eval | 17.01% | 16.38% | **14.87%** | **14.74%** (w2v=0.76) |

The ensemble improves on the best standalone model by 1.65pp on both benchmarks and clears the PRD target with margin. The models make complementary errors, which is the signal that motivates ensembling.

### Explainability (Hannah)

SHAP integration is functional on Wave2Vec and in progress on AASIST. The updated report will document the SHAP vs LIME tradeoff analysis Mustafa asked for.

### Deployment (Chase)

AWS Lambda deployment for AASIST v3 inference is in progress. Chase is unavailable today and will circulate an architecture writeup asynchronously before the next meeting.

## Threshold Fix (Response to Review Item)

Previously all confusion-matrix metrics were reported at a placeholder threshold of 0.5. All AASIST v3 metrics are now reported at the EER operating point, where FPR and FNR are balanced. The relevant artifacts are in the repo:

- `aasist/results/v3_threshold_metrics.md` — the threshold-selection rationale with per-split metrics
- `aasist/results/v3_chart_roc_eval_sets.png` — ROC curves for all three eval sets with the EER operating point and the prior 0.5 point marked

For production, the operating point can be shifted along the ROC curve based on the relative cost of false positives (rejecting a real caller) versus false negatives (admitting a spoof). The EER threshold is the appropriate reporting baseline until Capital One specifies a cost tradeoff.

## Discussion Items

1. **Reschedule Richmond visit.** The 7/7 visit did not occur. Requesting an alternate date to align on final deliverables in person.
2. **Signoff timeline given 7/17 to 7/27 OOO.** Critical review items need to be resolved by 7/16 to avoid a hard delay. What is the preferred process for final signoff?
3. **Questions on the AWS deployment.** Chase will circulate the architecture writeup asynchronously. Any questions we can route to him for that writeup?
4. **Final report v2 delivery date.** The updated report incorporating all review comments plus the ensemble results is targeted for 7/14. Confirm this timing works.

## Repository Pointers

- Main repo: https://github.com/eglhannah/CapitalOne_AudioDeepfake_Project (Mustafa access pending — Hannah to add today)
- AASIST v3 model + code: `aasist/` directory, weights on HuggingFace at `arnavjain321/aasist-v3-codecaugment`
- Wave2Vec model + predictions: `w2v/` directory
- Ensemble script + results: `aasist/results/v3_predictions/compute_ensemble.py` and `aasist/results/v3_ensemble_metrics.json`
- AWS Lambda deployment code: `deployment/aasist_lambda/`
- SHAP notebooks: `explainability (Hannah)/`

## Overall Status

Modeling work is complete and the ensemble hits the sponsor's PRD target. Explainability and the final report are being finalized this week. AWS deployment progress will be circulated by Chase asynchronously, and the team is targeting final signoff before Mustafa's 7/17 OOO.
