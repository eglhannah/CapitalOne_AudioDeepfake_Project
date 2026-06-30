#!/bin/bash
# v2 eval on ASVspoof 2019 LA dev split (in-domain reference)
set -e
cd "$(dirname "$0")"

CKPT="/scratch/$USER/aasist/outputs/runs/aasist_v2_rawboost_14642481/best.pt"
PROTO="/scratch/$USER/aasist/data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
AUDIO="/scratch/$USER/aasist/data/LA/ASVspoof2019_LA_dev/flac"
OUT="/scratch/$USER/aasist/outputs/eval/v2_2019_LA_dev"

python eval_aasist.py \
  --checkpoint "$CKPT" \
  --protocol "$PROTO" \
  --audio-root "$AUDIO" \
  --out-dir "$OUT" \
  --batch-size 32 \
  --num-workers 4
