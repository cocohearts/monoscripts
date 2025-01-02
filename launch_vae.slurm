#!/bin/bash
#SBATCH --job-name=accelerate_launch
#SBATCH --nodes=4
#SBATCH --gres=gpu:8             # Request 8 GPUs
#SBATCH --time=60:00:00          # Set a 60-hour time limit
#SBATCH --cpus-per-task=80
#SBATCH --partition=main  # Replace with the appropriate partition name
#SBATCH --output=slurm_logs/vae_%j.out
#SBATCH --error=slurm_logs/vae_%j.err

# Load necessary modules
# module load python
# module load cuda  # Or any other dependencies

# Activate your Python environment (if using one)
source ~/.venv/bin/activate  # Replace with your environment name if using Conda
export WANDB_MODE=online
export GPUS_PER_NODE=8
# Or use venv if applicable
# source /path/to/venv/bin/activate

echo "Allocated nodes: $SLURM_NODELIST"
srun echo "hey"

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Head node IP: $head_node_ip"

# Run your accelerate launch command
srun accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29503 \
    SD-VAE.py