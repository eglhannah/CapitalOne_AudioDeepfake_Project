#!/bin/bash
#SBATCH --job-name=latency_profile
#SBATCH --account=ds2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=/scratch/mhq8ka/aasist/logs/latency_profile_%j.out
#SBATCH --error=/scratch/mhq8ka/aasist/logs/latency_profile_%j.err

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

# Profile v2 checkpoint (the model going to production)
V2_CKPT="/scratch/$USER/aasist/outputs/runs/aasist_v2_rawboost_14642481/best.pt"

echo ""
echo "=========================================="
echo "  Latency profile — v2 on GPU"
echo "=========================================="
python latency_profile.py \
  --checkpoint "$V2_CKPT" \
  --out /scratch/$USER/aasist/outputs/latency/v2_gpu.json \
  --device cuda \
  --n-trials 50

echo ""
echo "=========================================="
echo "  Latency profile — v2 on CPU (production-like)"
echo "=========================================="
python latency_profile.py \
  --checkpoint "$V2_CKPT" \
  --out /scratch/$USER/aasist/outputs/latency/v2_cpu.json \
  --device cpu \
  --n-trials 20

echo ""
echo "=== Done ==="
date
