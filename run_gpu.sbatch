#!/bin/bash
#SBATCH --job-name=python-gpu
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --chdir=$SLURM_SUBMIT_DIR
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Create logs directory in the submission folder
mkdir -p logs

if [ $# -lt 1 ]; then
    echo "Usage: sbatch run_gpu.sbatch script.py [args]"
    exit 1
fi

SCRIPT="$1"
shift

echo "==============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working Dir: $(pwd)"
echo "Start Time: $(date)"
echo "==============================="

echo "GPU info:"
nvidia-smi || true

echo "Python version:"
python --version

python "$SCRIPT" "$@" 2>&1 | tee logs/run-${SLURM_JOB_ID}.log

echo "Finished at $(date)"