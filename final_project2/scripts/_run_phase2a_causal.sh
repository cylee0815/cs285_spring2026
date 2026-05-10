#!/usr/bin/env bash
# Phase 2A causal sanity check: re-run BC, AWAC, CQL, IQL on the
# leak-fixed compute_features pipeline.
#
# 12 runs: {bc, awac, cql_vanilla, iql} x {seed=42, 1337, 2024} x lambda=0.001.
# Output to results/phase2a_causal/ to keep separate from leaky Phase 2A.
#
# n_offline_updates = 20_000 (matches original Phase 2A).
# 4-way parallel batches.
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/logs/phase2a_causal"
mkdir -p "$LOG_DIR" "$PROJ_ROOT/results/phase2a_causal"

run_offline() {
    local algo="$1"
    local seed="$2"
    local tc="$3"
    local name="${algo}_lambda${tc}_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    if [ -f "$PROJ_ROOT/results/phase2a_causal/$name/metrics.json" ]; then
        echo "[skip] $name (metrics.json already exists)"
        return 0
    fi
    echo "[start] $name -> $log"
    uv run python scripts/run_offline.py \
        --base_config "$algo" \
        --seed "$seed" \
        --transaction_cost "$tc" \
        --behavior_mix mixture \
        --run_group phase2a_causal \
        --no_wandb \
        --n_offline_updates 20000 \
        --eval_interval 1000 \
        --results_dir results/phase2a_causal \
        --run_name "$name" \
        > "$log" 2>&1
    echo "[done $?] $name"
}

start_t=$(date +%s)

# 12 jobs total; mix algos within batches so any single-algo pathology
# doesn't monopolize the GPU.
JOBS=(
    "bc 42 0.001"          "bc 1337 0.001"        "bc 2024 0.001"        "awac 42 0.001"
    "awac 1337 0.001"      "awac 2024 0.001"      "cql_vanilla 42 0.001" "cql_vanilla 1337 0.001"
    "cql_vanilla 2024 0.001" "iql 42 0.001"       "iql 1337 0.001"       "iql 2024 0.001"
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

echo "===== Phase 2A causal complete in $(( $(date +%s) - start_t ))s ====="
