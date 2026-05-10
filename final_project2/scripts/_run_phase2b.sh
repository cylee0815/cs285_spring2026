#!/usr/bin/env bash
# Phase 2B online baselines at lambda=0.001, 100k env steps.
# Three batches of three jobs each so the H100 isn't contended by nine
# concurrent torch processes. Batch order: SAC -> PPO-LSTM -> GRPO.
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/logs/phase2b"
mkdir -p "$LOG_DIR" "$PROJ_ROOT/results/phase2b"

run_baseline() {
    local algo="$1"
    local seed="$2"
    local name="${algo}_lambda0.001_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    echo "[start] $name -> $log"
    uv run python scripts/run_online_baselines.py \
        --algo "$algo" --seed "$seed" \
        --dataset datasets/real_dirichlet.npz \
        --transaction_cost 0.001 \
        --total_timesteps 100000 \
        --eval_interval 10000 \
        --results_dir results/phase2b \
        --run_name "$name" \
        > "$log" 2>&1
    echo "[done $?] $name"
}

run_grpo() {
    local seed="$1"
    local name="grpo_lambda0.001_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    echo "[start] $name -> $log"
    uv run python scripts/train_grpo.py \
        --seed "$seed" \
        --dataset datasets/real_dirichlet.npz \
        --transaction_cost 0.001 \
        --total_env_steps 100000 \
        --log_every 5000 \
        --output_dir results/phase2b \
        --run_name "$name" \
        > "$log" 2>&1
    echo "[done $?] $name"
}

batch_start=$(date +%s)

echo "===== Batch 1/3: SAC-Dirichlet x 3 seeds ====="
run_baseline sac_dirichlet 42   &
run_baseline sac_dirichlet 1337 &
run_baseline sac_dirichlet 2024 &
wait
echo "[batch1] elapsed $(( $(date +%s) - batch_start ))s"

echo "===== Batch 2/3: PPO-LSTM x 3 seeds ====="
b2=$(date +%s)
run_baseline ppo_lstm 42   &
run_baseline ppo_lstm 1337 &
run_baseline ppo_lstm 2024 &
wait
echo "[batch2] elapsed $(( $(date +%s) - b2 ))s"

echo "===== Batch 3/3: GRPO x 3 seeds ====="
b3=$(date +%s)
run_grpo 42   &
run_grpo 1337 &
run_grpo 2024 &
wait
echo "[batch3] elapsed $(( $(date +%s) - b3 ))s"

echo "===== Phase 2B complete in $(( $(date +%s) - batch_start ))s ====="
