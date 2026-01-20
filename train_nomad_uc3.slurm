#!/bin/bash
#SBATCH --job-name=nomad_train
#SBATCH --partition=gpu_a100_il          # ggf. gpu_h100 / gpu_mi300 etc.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

# (Optional) Debug/Infos
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

# Module/Conda (Beispiele – je nach UC3 Setup)
module purge
# Häufig: Miniforge/Conda Modul (wenn bei euch verfügbar)
# module load devel/miniforge

# Wenn du bereits ein conda env hast:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nomad_train

# Headless-safe (Qt/OpenCV)
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

# (Optional) W&B offline falls outbound eingeschränkt:
# export WANDB_MODE=offline

cd $SLURM_SUBMIT_DIR

python train_adjusted.py -c config/nomad.yaml
