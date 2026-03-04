#!/bin/bash
#SBATCH --job-name=python-gpu
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --chdir=.
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Always execute from the directory where you ran sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Use the existing logs folder in the project root
LOGDIR="$PWD/logs"
mkdir -p "$LOGDIR"

if [ $# -lt 1 ]; then
  echo "Usage: sbatch run_gpu.sbatch script.py [args]"
  exit 1
fi

SCRIPT="$1"
shift

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpt

echo "==============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working Dir: $(pwd)"
echo "Start Time: $(date)"
echo "Script: $SCRIPT"
echo "Args: $@"
echo "==============================="

echo "GPU info:"
nvidia-smi || true

echo "Python version:"
python --version

python "$SCRIPT" "$@" 2>&1 | tee "$LOGDIR/run-${SLURM_JOB_ID}.log"

echo "Finished at $(date)"