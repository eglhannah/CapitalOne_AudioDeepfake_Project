#!/bin/bash
#SBATCH --job-name=aasist_fast
# Capstone allocation set up by Daniel Graham (DS 6015). Verify name with `allocations` on Rivanna; personal fallback: mhq8ka_rivanna
#SBATCH --account=ds2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/aasist_fast_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/aasist_fast_%j.err

set -e

module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh
conda activate aasist

echo "=== Job context ==="
date
hostname
nvidia-smi | head -15
echo "==="

cd /scratch/$USER/aasist/code/aasist_branch
python train_aasist.py --epochs 25 --batch-size 24 --num-workers 4 --run-name aasist_fast_${SLURM_JOB_ID}

echo "=== Done ==="
date
