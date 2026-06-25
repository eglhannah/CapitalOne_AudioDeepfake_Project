#!/bin/bash
#SBATCH --job-name=v3_2019_eval
#SBATCH --account=ds2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/v3_2019_eval_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/v3_2019_eval_%j.err

set -e

module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate aasist

echo "=== Job context ==="
date
hostname
which python
nvidia-smi | head -10
echo "==="

: "${V3_RUN_NAME:?Set V3_RUN_NAME=aasist_v3_codecreal_<train_job_id> when submitting}"

CKPT="/scratch/$USER/aasist/outputs/runs/${V3_RUN_NAME}/best.pt"
PROTO_DIR="/scratch/$USER/aasist/data/LA/ASVspoof2019_LA_cm_protocols"
test -f "$CKPT" || { echo "ERROR: checkpoint missing: $CKPT" >&2; exit 2; }

cd /scratch/$USER/aasist/code/aasist_branch

echo "=== 2019 LA dev ==="
python eval_aasist.py \
  --checkpoint "$CKPT" \
  --protocol "$PROTO_DIR/ASVspoof2019.LA.cm.dev.trl.txt" \
  --audio-root "/scratch/$USER/aasist/data/LA/ASVspoof2019_LA_dev/flac" \
  --out-dir "/scratch/$USER/aasist/outputs/eval/${V3_RUN_NAME}_2019_LA_dev" \
  --batch-size 32 \
  --num-workers 4

echo "=== 2019 LA eval ==="
python eval_aasist.py \
  --checkpoint "$CKPT" \
  --protocol "$PROTO_DIR/ASVspoof2019.LA.cm.eval.trl.txt" \
  --audio-root "/scratch/$USER/aasist/data/LA/ASVspoof2019_LA_eval/flac" \
  --out-dir "/scratch/$USER/aasist/outputs/eval/${V3_RUN_NAME}_2019_LA_eval" \
  --batch-size 32 \
  --num-workers 4

echo "=== Done ==="
date
