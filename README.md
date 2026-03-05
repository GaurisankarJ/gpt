
# ALICE GPU + Conda + Slurm Cheat Sheet

This is a practical, quick reference for working on the **ALICE HPC cluster** with **Python (conda)** and **A100 GPUs**.

---

# 1. Connect to ALICE

If using **eduVPN**, connect directly.

```bash
ssh alice
````

Example SSH config (`~/.ssh/config`):

```ssh
Host alice
    HostName login.alice.universiteitleiden.nl
    User s4374886
    IdentityFile ~/.ssh/id_rsa_alice
```

---

# 2. Useful SLURM Commands

### Show cluster partitions

```bash
sinfo
```

### Show your jobs

```bash
squeue -u $USER
```

### Show estimated start time

```bash
squeue --start -j JOBID
```

### Detailed job info

```bash
scontrol show job JOBID
```

### Cancel job

```bash
scancel JOBID
```

---

# 3. Load Conda Environment

Always load modules first.

```bash
module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
```

Activate conda:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpt
```

---

# 4. Create Python 3.11 Environment

```bash
conda create -n gpt -c conda-forge python=3.11 -y
conda activate gpt
```

Check version:

```bash
python --version
```

---

# 5. Interactive GPU Session (Debugging)

Get an A100 GPU shell:

```bash
srun \
--partition=gpu-a100-80g \
--gres=gpu:1 \
--cpus-per-task=8 \
--mem=80G \
--time=04:00:00 \
--pty bash
```

Check GPU:

```bash
nvidia-smi
```

Run your script:

```bash
python train.py
```

---

# 6. Batch Job Submission

Submit a job with Slurm:

```bash
sbatch run_gpu.sbatch train.py
```

With arguments:

```bash
sbatch run_gpu.sbatch train.py --lr 0.001 --epochs 10
```

---

# 7. Generic GPU Runner Script

`run_gpu.sbatch`

```bash
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
```

---

# 8. Check Job Logs

Slurm logs:

```bash
logs/python-gpu-JOBID.out
logs/python-gpu-JOBID.err
```

Python runtime logs:

```bash
logs/run-JOBID.log
```

Watch logs live:

```bash
tail -f logs/run-JOBID.log
```

---

# 9. Check GPU Usage

```bash
nvidia-smi
```

Continuous monitoring:

```bash
watch -n 2 nvidia-smi
```

---

# 10. Useful Debugging

Check environment:

```bash
which python
python --version
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 11. Quick Dev Workflow

Typical development loop:

```bash
ssh alice
cd project

srun --partition=gpu-a100-80g --gres=gpu:1 --pty bash

conda activate gpt
python train.py
```

For long jobs:

```bash
sbatch run_gpu.sbatch train.py
```

---

# 12. Folder Structure (Recommended)

```
project/
│
├── run_gpu.sbatch
├── logs/
├── data/
└── checkpoints/
```

---

# 13. Helpful Commands

### show partitions

```bash
sinfo
```

### see GPU queues

```bash
squeue -p gpu-a100-80g
```

### list modules

```bash
module avail
```

### search modules

```bash
module spider python
```

---

# 14. Common Job States

|Code|Meaning|
|---|---|
|PD|Pending (waiting in queue)|
|R|Running|
|CG|Completing|
|F|Failed|
|CD|Completed|

---

# 15. Monitor Job Progress

```bash
squeue -u $USER
```

Watch job logs:

```bash
tail -f logs/run-JOBID.log
```

---

# 16. Cancel Jobs

Cancel one job:

```bash
scancel JOBID
```

Cancel all your jobs:

```bash
scancel -u $USER
```

---

# 17. Best Practices

✔ Never run heavy code on **login nodes**  
✔ Use **srun for debugging**  
✔ Use **sbatch for training jobs**  
✔ Always log output  
✔ Keep experiments reproducible

---

# 18. Quick GPU Job Template

```bash
sbatch run_gpu.sbatch train.py --epochs 10
```

---

# 19. Copy Files Between Local and ALICE

### Local to ALICE

Copy a local checkpoint file to the remote `alice` machine:

```bash
scp ./checkpoints/qwen3_0.6b.pth alice:~/omega/gpt/checkpoints
```

### ALICE to Local

Sync file from `alice` to your local `./logs` folder:

```bash
rsync -avz alice:~/omega/gpt/checkpoints/qwen3_0.6b.pth ./checkpoints
```

Copy a single checkpoint from `alice` to local:

```bash
scp alice:~/omega/gpt/checkpoints/qwen3_0.6b.pth ./checkpoints
```
