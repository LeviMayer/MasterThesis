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

# ---- Modules ----
module purge
module load devel/miniforge

# ---- Headless safe (Qt/OpenCV/Matplotlib) ----
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

# ---- CUDA memory fragmentation mitigation ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Weights & Biases (recommended on clusters) ----
export WANDB_MODE=offline
export WANDB_DIR="$SLURM_SUBMIT_DIR/wandb"
# If you want to disable completely, use instead:
# export WANDB_DISABLED=true

# ---- Activate env ----
conda activate nomad_train

# ---- Sanity checks ----
nvidia-smi || true
python - <<'PY'
import sys
print("Python:", sys.executable)
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check failed:", repr(e))
PY

# ---- Run training from submit dir ----
cd "$SLURM_SUBMIT_DIR"

python train_adjusted.py -c config/nomad.yaml
