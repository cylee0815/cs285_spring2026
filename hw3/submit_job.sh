#!/bin/bash
#SBATCH --job-name=hw3_rl_training
#SBATCH --output=training_output.log # Where print statements will go
#SBATCH --partition=GPU-shared       # Request a GPU partition
#SBATCH --gpus=2                     # Request 2 GPU
#SBATCH --time=8:00:00              # Max time limit (12 hours)

# Navigate to your homework directory
cd /jet/home/tlee8/cs285_spring2026/hw3

# Run your bash script or python commands
./run.sh
