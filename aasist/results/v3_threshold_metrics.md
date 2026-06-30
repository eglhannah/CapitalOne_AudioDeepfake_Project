# AASIST v3 metrics at the EER operating point

**Context.** The 6/17 progress report reported all confusion-matrix-based metrics at threshold = 0.5, noting that 'our team has not determined a suitable ruling method.' This file replaces that placeholder with metrics computed at the EER operating point, the threshold where FPR and FNR are equal. EER is the standard operating point for anti-spoofing systems where bonafide and spoof errors are equally costly.

## Per-split summary

| Eval split | n | AUC | EER | EER threshold | Acc @ EER thr | Acc @ 0.5 (prior) |
|---|---|---|---|---|---|---|
| 2019 LA dev | 24,844 | 0.9997 | 0.76% | 0.5820 | 99.26% | 99.38% |
| 2019 LA eval | 71,237 | 0.9981 | 1.67% | 0.1006 | 98.33% | 95.77% |
| 2021 LA eval | 148,176 | 0.9880 | 4.67% | 0.8075 | 95.33% | 97.16% |
| 2021 DF eval | 533,928 | 0.9227 | 17.01% | 0.8425 | 82.99% | 88.71% |

## Why this matters

At threshold = 0.5 the system applies a uniform decision rule that does not account for the score distribution learned by AASIST v3. On 2021 DF the default threshold misclassifies a larger share of bonafide audio than necessary (high FPR), and on 2019 LA dev the default underuses the model's headroom (FPR and FNR are far from balanced). The EER threshold gives a single defensible operating point that is reproducible per split and comparable across the team's models.

## For production deployment

EER is the *symmetric* operating point. For a production fraud system, the operating point should be tuned to the relative cost of FPR (rejecting a real customer) versus FNR (admitting a spoofed call). If a fraud-loss-versus-customer-friction tradeoff is provided by the sponsor, the same ROC curves below allow selection of any operating point along the curve. Until that tradeoff is specified, EER is the appropriate reporting threshold.

## Generated artifacts

- `v3_threshold_metrics.json` - full confusion matrices at both thresholds
- `v3_chart_roc_eval_sets.png` - ROC curves for the three external eval sets, with the EER point and the 0.5 point marked
