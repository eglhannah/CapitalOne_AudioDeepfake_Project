#!/bin/bash
#SBATCH --job-name=codec_pre
#SBATCH --account=ds2022
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/codec_precompute_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/codec_precompute_%j.err

set -e

module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate aasist

echo "=== Job context ==="
date
hostname
echo "Job ID: $SLURM_JOB_ID"
which ffmpeg
ffmpeg -version | head -1
echo "==="

cd /scratch/$USER/aasist/code/aasist_branch
python precompute_codec_train.py --workers 16

echo "=== Done ==="
date
