#!/bin/bash
#SBATCH --job-name=accelerate_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8             # Request 8 GPUs
#SBATCH --time=60:00:00          # Set a 60-hour time limit
#SBATCH --cpus-per-task=80
#SBATCH --partition=main  # Replace with the appropriate partition name
#SBATCH --output=accelerate_launch_%j.out
#SBATCH --error=accelerate_launch_%j.err

# Load necessary modules
# module load python
# module load cuda  # Or any other dependencies

# Activate your Python environment (if using one)
source ~/.venv/bin/activate  # Replace with your environment name if using Conda
export WANDB_MODE=online
# Or use venv if applicable
# source /path/to/venv/bin/activate

echo "Allocated nodes: $SLURM_NODELIST"
srun echo "hey"

# Run your accelerate launch command
srun accelerate launch mamba.py