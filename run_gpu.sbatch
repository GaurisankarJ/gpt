#!/bin/bash
#SBATCH --job-name=python-gpu
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Create logs directory if missing
mkdir -p logs

# check argument
if [ $# -lt 1 ]; then
    echo "Usage: sbatch run_gpu.sbatch script.py [args]"
    exit 1
fi

SCRIPT="$1"
shift

echo "==============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Working Dir: $(pwd)"
echo "Script: $SCRIPT"
echo "Args: $@"
echo "==============================="

echo "CPU info:"
lscpu | head -n 10 || true

echo "GPU info:"
nvidia-smi || true

echo "Python version:"
python --version

echo "==============================="
echo "Starting Python script"
echo "==============================="

# Run script and stream output
python "$SCRIPT" "$@" 2>&1 | tee logs/run-${SLURM_JOB_ID}.log

echo "==============================="
echo "Job finished at: $(date)"
echo "==============================="