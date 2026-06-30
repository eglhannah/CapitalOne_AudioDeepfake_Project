#!/bin/bash
#SBATCH --job-name=aasist_full
# Capstone allocation set up by Daniel Graham (DS 6015). Verify name with `allocations` on Rivanna; personal fallback: mhq8ka_rivanna
#SBATCH --account=ds2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/aasist_full_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/aasist_full_%j.err

set -e

# Load env
module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh
conda activate aasist

# Print context
echo "=== Job context ==="
date
hostname
nvidia-smi | head -15
echo "==="

# Run
cd /scratch/$USER/aasist/code/aasist_branch
python train_aasist.py --epochs 50 --batch-size 24 --num-workers 4 --run-name aasist_full_${SLURM_JOB_ID}

echo "=== Done ==="
date
