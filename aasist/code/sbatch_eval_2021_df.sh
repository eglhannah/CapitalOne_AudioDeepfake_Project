#!/bin/bash
#SBATCH --job-name=df_eval
# Capstone allocation set up by Daniel Graham (DS 6015). Verify name with `allocations` on Rivanna; personal fallback: mhq8ka_rivanna
#SBATCH --account=ds2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/df_eval_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/df_eval_%j.err

set -e

# Load env
module load miniforge/26.1.0
source /apps/software/standard/core/miniforge/26.1.0/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate aasist

# Print context
echo "=== Job context ==="
date
hostname
echo "Job ID: $SLURM_JOB_ID"
which python
nvidia-smi | head -15
echo "==="

# Run the eval
cd /scratch/$USER/aasist/code/aasist_branch
bash run_eval_2021_df.sh

echo "=== Done ==="
date
