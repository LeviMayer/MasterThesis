#!/bin/bash
#SBATCH --job-name=nomad_train
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail
mkdir -p logs

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PWD:  $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

module purge
module load devel/miniforge

# Headless-safe
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B offline (optional)
export WANDB_MODE=offline
export WANDB_DIR="$SLURM_SUBMIT_DIR/wandb"

# Use the conda executable directly (no activate/deactivate)
CONDA_EXE="/opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/bin/conda"

nvidia-smi || true

# Sanity check
"$CONDA_EXE" run -n nomad_train --no-capture-output python - <<'PY'
import sys
print("Python:", sys.executable)
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

cd "$SLURM_SUBMIT_DIR"

# Training
"$CONDA_EXE" run -n nomad_train --no-capture-output python train_adjusted.py -c config/nomad.yaml
