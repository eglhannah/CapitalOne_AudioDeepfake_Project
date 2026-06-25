#!/bin/bash
#SBATCH --job-name=v3_df_eval
#SBATCH --account=ds2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/v3_df_eval_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/v3_df_eval_%j.err

set -e

module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate aasist

echo "=== Job context ==="
date
hostname
which python
nvidia-smi | head -15
echo "==="

: "${V3_RUN_NAME:?Set V3_RUN_NAME=aasist_v3_codecreal_<train_job_id> when submitting}"

CKPT="/scratch/$USER/aasist/outputs/runs/${V3_RUN_NAME}/best.pt"
META="/scratch/$USER/aasist/data/2021/keys/DF/CM/trial_metadata.txt"
AUDIO="/scratch/$USER/aasist/data/2021/ASVspoof2021_DF_eval/flac"
OUT="/scratch/$USER/aasist/outputs/eval/${V3_RUN_NAME}_2021_DF_eval"

test -f "$CKPT" || { echo "ERROR: checkpoint missing: $CKPT" >&2; exit 2; }

cd /scratch/$USER/aasist/code/aasist_branch
python eval_aasist_2021_df.py \
  --checkpoint "$CKPT" \
  --metadata "$META" \
  --audio-root "$AUDIO" \
  --out-dir "$OUT" \
  --batch-size 32 \
  --num-workers 4

echo "=== Done ==="
date
