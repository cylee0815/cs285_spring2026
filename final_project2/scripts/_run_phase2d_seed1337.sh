#!/usr/bin/env bash
# Phase 2D second seed: G ∈ {4, 8, 16} at seed=1337 to add variance bars
# to the inverse-G Sharpe finding observed at seed=42.
# Mirrors scripts/_run_phase2d_grpo_ablation.sh exactly except for seed.
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/logs/phase2d"
mkdir -p "$LOG_DIR" "$PROJ_ROOT/results/phase2d"

run_grpo() {
    local g="$1"
    local seed=1337
    local name="grpo_G${g}_lambda0.001_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    if [ -f "$PROJ_ROOT/results/phase2d/$name/metrics.json" ]; then
        echo "[skip] $name (metrics.json already exists)"
        return 0
    fi
    echo "[start] $name -> $log"
    uv run python scripts/train_grpo.py \
        --seed "$seed" \
        --dataset datasets/real_dirichlet.npz \
        --transaction_cost 0.001 \
        --total_env_steps 100000 \
        --group_size "$g" \
        --log_every 5000 \
        --output_dir results/phase2d \
        --run_name "$name" \
        > "$log" 2>&1
    echo "[done $?] $name"
}

start_t=$(date +%s)

# 3-way parallel — three GRPO processes fit easily on the H100.
run_grpo 4  &
run_grpo 8  &
run_grpo 16 &
wait

echo "===== Phase 2D seed=1337 complete in $(( $(date +%s) - start_t ))s ====="
