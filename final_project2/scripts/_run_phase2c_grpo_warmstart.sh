#!/usr/bin/env bash
# Phase 2C "GRPO with offline warm-start" condition.
# Three seeds, all warm-started from the SAME source checkpoint
# (causal Phase 2A IQL seed=42, which the smoke test confirmed retains
# observation-dependent structure with weight-std 0.0727 across states).
#
# Output: results/phase2c/grpo_warm_lambda0.001_seed{42,1337,2024}/
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/logs/phase2c"
mkdir -p "$LOG_DIR" "$PROJ_ROOT/results/phase2c"

CKPT="$PROJ_ROOT/results/phase2a_causal/iql_lambda0.001_seed42/actor.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: source checkpoint $CKPT not found" >&2
    exit 1
fi

run_grpo_warm() {
    local seed="$1"
    local name="grpo_warm_lambda0.001_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    if [ -f "$PROJ_ROOT/results/phase2c/$name/metrics.json" ]; then
        echo "[skip] $name (metrics.json already exists)"
        return 0
    fi
    echo "[start] $name -> $log  (warm-start from $CKPT)"
    uv run python scripts/train_grpo.py \
        --seed "$seed" \
        --dataset datasets/real_dirichlet.npz \
        --transaction_cost 0.001 \
        --total_env_steps 100000 \
        --group_size 4 \
        --log_every 5000 \
        --output_dir results/phase2c \
        --run_name "$name" \
        --init_actor_checkpoint "$CKPT" \
        > "$log" 2>&1
    echo "[done $?] $name"
}

start_t=$(date +%s)
# 3-way parallel — three GRPO processes fit easily on the H100 alongside
# the in-flight Phase 2C O2O runs (which are the SAC fine-tune path,
# different model class).
run_grpo_warm 42   &
run_grpo_warm 1337 &
run_grpo_warm 2024 &
wait
echo "===== Phase 2C GRPO warm-start complete in $(( $(date +%s) - start_t ))s ====="
