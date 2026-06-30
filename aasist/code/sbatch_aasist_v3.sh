#!/bin/bash
#SBATCH --job-name=aasist_v3
#SBATCH --account=ds2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/aasist_v3_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/aasist_v3_%j.err

set -e

module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate aasist

echo "=== Job context ==="
date
hostname
echo "Job ID: $SLURM_JOB_ID"
which python
nvidia-smi | head -15
echo "==="

# Sanity: pre-computed codec cache must exist
test -d /scratch/$USER/aasist/data/LA_codec_train/alaw || {
    echo "ERROR: codec cache missing. Run sbatch sbatch_precompute_codec_train.sh first." >&2
    exit 2
}

cd /scratch/$USER/aasist/code/aasist_branch
python train_aasist_v3.py \
  --epochs 25 \
  --batch-size 24 \
  --num-workers 4 \
  --p-codec 0.5 \
  --codecs alaw ulaw g722 opus \
  --run-name aasist_v3_codecreal_${SLURM_JOB_ID}

echo "=== Done ==="
date
