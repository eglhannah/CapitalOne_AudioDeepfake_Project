#!/bin/bash
# Run AASIST eval on ASVspoof 2021 DF eval split
# (unknown attacks + 100+ TTS/VC systems + media compression conditions)
set -e
cd "$(dirname "$0")"

CKPT="/scratch/$USER/aasist/outputs/runs/aasist_fast_13608120/best.pt"
META="/scratch/$USER/aasist/data/2021/keys/DF/CM/trial_metadata.txt"
AUDIO="/scratch/$USER/aasist/data/2021/ASVspoof2021_DF_eval/flac"
OUT="/scratch/$USER/aasist/outputs/eval/2021_DF_eval"

python eval_aasist_2021_df.py \
  --checkpoint "$CKPT" \
  --metadata "$META" \
  --audio-root "$AUDIO" \
  --out-dir "$OUT" \
  --batch-size 32 \
  --num-workers 4
