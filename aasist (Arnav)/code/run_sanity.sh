#!/bin/bash
# AASIST sanity check: 200 train / 200 dev samples, 5 epochs
set -e
cd "$(dirname "$0")"
python train_aasist.py \
  --limit-train 200 \
  --limit-dev 200 \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 2 \
  --run-name sanity_check
