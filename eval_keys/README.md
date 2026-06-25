# ASVspoof 2021 DF Evaluation Keys (canonical)

The official ASVspoof 2021 DF eval keys file, mirrored here so the team can join predictions against a single canonical source without each person re-downloading from the ASVspoof site.

## Files

| File | Size | What |
|---|---|---|
| `ASVspoof2021_DF_keys.csv` | 22 MB | Slim CSV with the four columns most evaluators need: `file_id, label, attack_id, compression` |
| `ASVspoof2021_DF_trial_metadata.txt.gz` | 6 MB | Full original keys file from the ASVspoof2021 release, gzipped. Has 13 columns (speaker_id, file_id, compression, source_corpus, attack_id, label, trim, partition, vocoder_family, task, team, channel, language). Use this if you need columns beyond what the slim CSV exposes. |

## Counts

- Total trials: 611,829
- Bonafide: 22,617
- Spoof: 589,212

## Schemas

**Slim CSV columns:**

```
file_id            DF_E_xxxxxxx          (join key)
label              bonafide | spoof      (target)
attack_id          A07..A19 | Task{1,2}-team{NN} | HUB-Bxx | SPO-Nxx | -
compression        nocodec | low_mp3 | high_mp3 | low_m4a | high_m4a | mp3m4a | oggm4a | low_ogg | high_ogg
```

For bonafide rows, `attack_id` is `-` (a literal dash).

**Full file columns (space-separated):**

```
speaker_id  file_id  compression  source_corpus  attack_id  label  trim  partition  vocoder_family  task  team  channel  language
```

## Usage

In pandas:

```python
import pandas as pd

keys = pd.read_csv("eval_keys/ASVspoof2021_DF_keys.csv")
predictions = pd.read_csv("your_predictions.csv")  # must have a file_id column

merged = predictions.merge(keys[["file_id", "label"]], on="file_id", how="inner", suffixes=("_pred", "_true"))
# Sanity: merged should have exactly 611,829 rows
assert len(merged) == 611_829, f"protocol mismatch: got {len(merged)}"
```

If your prediction CSV's true-label column differs from the canonical one for some rows, those are the protocol-discrepancy files. Re-derive your `true_label` column from `keys.label` and recompute EER.

## Provenance

Sourced from Rivanna at `/scratch/$USER/aasist/data/2021/keys/DF/CM/trial_metadata.txt` (file mtime 2022-10-01 — this is the post-competition full eval keys release from ASVspoof2021).
