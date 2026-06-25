# AASIST v3 predictions

Per-utterance scores from the v3 AASIST checkpoint (best epoch 19, dev EER 0.78%).

**Training recipe:** CodecAugment (real telephony codec passes: alaw, ulaw, g722, opus, p=0.5) + RawBoost noise-only (lnl/isd/ssi, codec channel removed since CodecAugment covers it).

**Checkpoint:** `/scratch/$USER/aasist/outputs/runs/aasist_v3_codecreal_16462819/best.pt` on Rivanna.

## Files

| File | Rows | EER | AUC | Acc |
|---|---|---|---|---|
| `v3_2019_LA_dev_predictions.csv` | 24,844 | 0.78% | 0.9997 | 99.4% |
| `v3_2019_LA_eval_predictions.csv` | 71,237 | 1.67% | 0.9981 | 95.8% |
| `v3_2021_LA_eval_predictions.csv` | 148,176 | 4.67% | 0.9880 | 97.2% |
| `v3_2021_DF_eval_predictions.csv` | 533,928 | 17.00% | 0.9227 | 88.7% |

Each `v3_*_summary.json` has the full breakdown (per-attack EER, per-codec/per-compression EER, confusion matrix, threshold info).

## CSV schema

```
utterance_id    LA_D_xxxxxxx | LA_E_xxxxxxx | DF_E_xxxxxxx
label           0 = bonafide, 1 = spoof
score           model output ∈ [0, 1] (higher = more likely spoof)
```

Note: label is `0/1` here, not the string `bonafide/spoof` used in the canonical eval keys (`eval_keys/ASVspoof2021_DF_keys.csv`). Map `0 → bonafide, 1 → spoof` when joining.

## 2021 DF row count: 533,928 not 611,829

The full ASVspoof 2021 DF eval protocol has 611,829 trials, but the public-release evaluation phase is the 533,928-trial subset (everything except the hidden/test partition that ASVspoof never released scoring for). All published 2021 DF EER numbers are on this subset.

## Recompute EER

```python
import pandas as pd
from sklearn.metrics import roc_curve

df = pd.read_csv("v3_2021_DF_eval_predictions.csv")
fpr, tpr, _ = roc_curve(df.label, df.score)
fnr = 1 - tpr
eer = fnr[(fnr - fpr).abs().argmin()]
print(f"EER: {eer:.4f}")
```

## Comparison

| Eval | v1 (no aug) | v2 (RawBoost) | **v3 (Codec+RB-noise)** |
|---|---|---|---|
| 2019 LA eval | 3.33% | 1.87% | **1.67%** |
| 2021 LA eval | 5.67% | 8.01% | **4.67%** |
| 2021 DF eval | 22.95% | 17.20% | **17.00%** |

v3 is best on every benchmark. Biggest win is 2021 LA (~42% relative reduction vs v2) — confirms the CodecAugment thesis (real telephony codecs at train time generalize to the 2021 LA channel distribution).
