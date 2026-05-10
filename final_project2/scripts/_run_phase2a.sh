#!/usr/bin/env bash
# Phase 2A offline matrix at lambda=0.001, plus IQL lambda-anchor at {0,0.001,0.005}.
# 24 runs total: {bc, td3_bc, cql_vanilla, awac, bcq} x lambda=0.001 x 3 seeds = 15
# plus iql x {0, 0.001, 0.005} x 3 seeds = 9.
#
# n_offline_updates = 20_000 (config default is 100k; the milestone showed
# val-Sharpe peaks at ~step 1000 and decays, so 20k captures peak + decay
# with a 5x wall-clock saving).
#
# 4-way parallel batches.
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/logs/phase2a"
mkdir -p "$LOG_DIR" "$PROJ_ROOT/results/phase2a"

run_offline() {
    local algo="$1"
    local seed="$2"
    local tc="$3"
    local name="${algo}_lambda${tc}_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    if [ -f "$PROJ_ROOT/results/phase2a/$name/metrics.json" ]; then
        echo "[skip] $name (metrics.json already exists)"
        return 0
    fi
    echo "[start] $name -> $log"
    uv run python scripts/run_offline.py \
        --base_config "$algo" \
        --seed "$seed" \
        --transaction_cost "$tc" \
        --behavior_mix mixture \
        --run_group phase2a \
        --no_wandb \
        --n_offline_updates 20000 \
        --eval_interval 1000 \
        --results_dir results/phase2a \
        --run_name "$name" \
        > "$log" 2>&1
    echo "[done $?] $name"
}

start_t=$(date +%s)

# Batch jobs into groups of 4 for the H100. Order them by algo so logs are
# readable; mixing seeds within a batch keeps any single-seed pathology
# from monopolizing the GPU.
JOBS=(
    "bc 42 0.001"          "bc 1337 0.001"        "bc 2024 0.001"        "td3_bc 42 0.001"
    "td3_bc 1337 0.001"    "td3_bc 2024 0.001"    "cql_vanilla 42 0.001" "cql_vanilla 1337 0.001"
    "cql_vanilla 2024 0.001" "awac 42 0.001"      "awac 1337 0.001"      "awac 2024 0.001"
    "bcq 42 0.001"         "bcq 1337 0.001"       "bcq 2024 0.001"       "iql 42 0"
    "iql 1337 0"           "iql 2024 0"           "iql 42 0.001"         "iql 1337 0.001"
    "iql 2024 0.001"       "iql 42 0.005"         "iql 1337 0.005"       "iql 2024 0.005"
)

PARALLELISM=4
total=${#JOBS[@]}
batch_idx=1
n_batches=$(( (total + PARALLELISM - 1) / PARALLELISM ))

for (( i = 0; i < total; i += PARALLELISM )); do
    echo "===== Batch $batch_idx/$n_batches ====="
    bt=$(date +%s)
    for (( j = 0; j < PARALLELISM && i + j < total; j++ )); do
        spec=${JOBS[i+j]}
        # shellcheck disable=SC2086
        run_offline $spec &
    done
    wait
    echo "[batch$batch_idx] elapsed $(( $(date +%s) - bt ))s"
    batch_idx=$((batch_idx + 1))
done

echo "===== Phase 2A complete in $(( $(date +%s) - start_t ))s ====="
