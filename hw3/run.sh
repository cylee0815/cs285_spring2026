#!/bin/bash

# Exit immediately if a command exits with a non-zero status
# (Remove this line if you want the script to continue even if one run fails)
set -e

# echo "==================================================="
# echo "Starting Deep Q-Learning (DQN) Hyperparameter Sweeps"
# echo "==================================================="

# echo "Running LunarLander High LR..."
# uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_lr_high.yaml 

# echo "Running LunarLander Low LR..."
# uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_lr_low.yaml 

# echo "Running LunarLander Target Freq 500..."
# uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_freq_500.yaml 

# echo "Running LunarLander Target Freq 1500..."
# uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_freq_1500.yaml 

# echo "Running LunarLander Target Freq 5000..."
# uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_freq_5000.yaml 


echo "==================================================="
echo "Starting Soft Actor-Critic (SAC) Experiments"
echo "==================================================="

# echo "Running SAC Base (HalfCheetah)..."
# echo "WARNING: HalfCheetah is computationally expensive! This may take a while locally."
# uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml 

echo "Running SAC Autotune (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune.yaml 

# Section 3.6: Stabilizing Target Values (Clipped Double-Q)
echo "Running SAC Single-Q (Hopper)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_singleq.yaml 

echo "Running SAC Clipped Double-Q (Hopper)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_clipq.yaml 

echo "Running SAC 0.01 (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune_1e-2.yaml 
echo "Running SAC 0.05 (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune_5e-2.yaml 
echo "Running SAC 0.5 (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune_5e-1.yaml 
echo "Running SAC 1 (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune_1.yaml 
echo "Running SAC Autotune 0.01 (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune_1e-2.yaml 
echo "Running SAC Autotune 0.05 (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune_5e-2.yaml 
echo "Running SAC Autotune 0.5 (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune_5e-1.yaml 
echo "Running SAC Autotune 1 (HalfCheetah)..."
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune_1.yaml 

echo "==================================================="
echo "All experiments finished successfully!"
echo "==================================================="
