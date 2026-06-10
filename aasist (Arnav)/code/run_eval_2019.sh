#!/bin/bash
# Run AASIST eval on ASVspoof 2019 LA eval split (unknown attacks A07-A19)
set -e
cd "$(dirname "$0")"

CKPT="/scratch/$USER/aasist/outputs/runs/aasist_fast_13608120/best.pt"
PROTO="/scratch/$USER/aasist/data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
AUDIO="/scratch/$USER/aasist/data/LA/ASVspoof2019_LA_eval/flac"
OUT="/scratch/$USER/aasist/outputs/eval/2019_LA_eval"

python eval_aasist.py \
  --checkpoint "$CKPT" \
  --protocol "$PROTO" \
  --audio-root "$AUDIO" \
  --out-dir "$OUT" \
  --batch-size 32 \
  --num-workers 4
