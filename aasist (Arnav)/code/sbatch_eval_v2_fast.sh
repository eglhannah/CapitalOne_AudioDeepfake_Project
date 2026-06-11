#!/bin/bash
#SBATCH --job-name=v2_fast_evals
#SBATCH --account=ds2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/v2_fast_evals_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/v2_fast_evals_%j.err

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

cd /scratch/$USER/aasist/code/aasist_branch

echo ""
echo "=========================================="
echo "  EVAL 1/3 — 2019 LA dev (in-domain)"
echo "=========================================="
bash run_eval_v2_2019_dev.sh

echo ""
echo "=========================================="
echo "  EVAL 2/3 — 2019 LA eval (unknown attacks)"
echo "=========================================="
bash run_eval_v2_2019_eval.sh

echo ""
echo "=========================================="
echo "  EVAL 3/3 — 2021 LA (telephony codecs)"
echo "=========================================="
bash run_eval_v2_2021_la.sh

echo ""
echo "=== All 3 fast evals complete ==="
date
