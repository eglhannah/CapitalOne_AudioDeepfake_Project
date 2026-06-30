#!/bin/bash
# v2 eval on ASVspoof 2021 LA eval (unknown attacks + 7 telephony codecs)
set -e
cd "$(dirname "$0")"

CKPT="/scratch/$USER/aasist/outputs/runs/aasist_v2_rawboost_14642481/best.pt"
META="/scratch/$USER/aasist/data/2021/keys/LA/CM/trial_metadata.txt"
AUDIO="/scratch/$USER/aasist/data/2021/ASVspoof2021_LA_eval/flac"
OUT="/scratch/$USER/aasist/outputs/eval/v2_2021_LA_eval"

python eval_aasist_2021_la.py \
  --checkpoint "$CKPT" \
  --metadata "$META" \
  --audio-root "$AUDIO" \
  --out-dir "$OUT" \
  --batch-size 32 \
  --num-workers 4
