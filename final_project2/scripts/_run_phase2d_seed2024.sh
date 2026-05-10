#!/usr/bin/env bash
# Phase 2D third seed: G in {4, 8, 16} at seed=2024 to firm up variance bars
# (n=3 seeds matches Phase 2A/B convention). Mirrors the seed=1337 script.
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/logs/phase2d"
mkdir -p "$LOG_DIR" "$PROJ_ROOT/results/phase2d"

run_grpo() {
    local g="$1"
    local seed=2024
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
run_grpo 4  &
run_grpo 8  &
run_grpo 16 &
wait
echo "===== Phase 2D seed=2024 complete in $(( $(date +%s) - start_t ))s ====="
