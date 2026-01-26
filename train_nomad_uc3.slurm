#!/bin/bash
#SBATCH --job-name=nomad_train
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --export=NONE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levi.mayer@uni-ulm.de

set -euo pipefail
mkdir -p logs

# Minimal sane environment (weil export=NONE)
export PATH=/usr/bin:/bin
export HOME="$HOME"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PWD:  $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

# Absolutely remove any inherited conda markers (extra safety)
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL
unset _CE_CONDA _CE_M

module purge
module load devel/miniforge

CONDA_EXE="/opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/bin/conda"

# Headless safe
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B offline (optional)
export WANDB_MODE=offline
export WANDB_DIR="$SLURM_SUBMIT_DIR/wandb"

cd "$SLURM_SUBMIT_DIR"

nvidia-smi || true

# Sanity check
"$CONDA_EXE" run -n nomad_train --no-capture-output python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Run training
"$CONDA_EXE" run -n nomad_train --no-capture-output python train_adjusted.py -c config/nomad.yaml
