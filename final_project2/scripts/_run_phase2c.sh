#!/usr/bin/env bash
# Phase 2C O2O comparison at lambda=0.001.
# Conditions 1 (frozen offline) and 2 (online-only SAC) are free — the
# aggregator pulls them from Phase 2A and Phase 2B respectively.
# This script trains conditions 3 and 4: naive fine-tune and adaptive O2O,
# 3 seeds each = 6 runs.
#
# Run sizes:
#   n_offline_updates = 20_000  (matches Phase 2A)
#   n_online_steps    = 50_000  (half of Phase 2B's 100k to keep total 6-run
#                                wall-clock under 90 min at 2-way parallel —
#                                the absolute-step difference doesn't break
#                                the four-way comparison since Phase 2C is
#                                fine-tuning a pre-trained policy, not
#                                training from scratch)
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/logs/phase2c"
mkdir -p "$LOG_DIR" "$PROJ_ROOT/results/phase2c"

run_o2o() {
    local cond="$1"          # naive | adaptive
    local seed="$2"
    local adaptive_flag
    if [ "$cond" = "adaptive" ]; then
        adaptive_flag="true"
    else
        adaptive_flag="false"
    fi
    local name="${cond}_o2o_lambda0.001_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    if [ -f "$PROJ_ROOT/results/phase2c/$name/metrics.json" ]; then
        echo "[skip] $name (metrics.json already exists)"
        return 0
    fi
    echo "[start] $name -> $log"
    uv run python scripts/run_o2o.py \
        --phase=o2o \
        --seed="$seed" \
        --behavior_mix=mixture \
        --adaptive_conservatism="$adaptive_flag" \
        --transaction_cost=0.001 \
        --n_offline_updates=20000 \
        --n_online_steps=50000 \
        --offline_data_steps=50000 \
        --eval_interval=5000 \
        --episode_length=63 \
        --run_group=phase2c \
        --results_dir=results/phase2c \
        --run_name="$name" \
        > "$log" 2>&1
    echo "[done $?] $name"
}

start_t=$(date +%s)

PARALLELISM=2  # O2O runs are heavier than offline-only or online-only
JOBS=(
    "naive 42"     "naive 1337"    "naive 2024"
    "adaptive 42"  "adaptive 1337" "adaptive 2024"
)
total=${#JOBS[@]}
batch_idx=1
n_batches=$(( (total + PARALLELISM - 1) / PARALLELISM ))

for (( i = 0; i < total; i += PARALLELISM )); do
    echo "===== Batch $batch_idx/$n_batches ====="
    bt=$(date +%s)
    for (( j = 0; j < PARALLELISM && i + j < total; j++ )); do
        spec=${JOBS[i+j]}
        # shellcheck disable=SC2086
        run_o2o $spec &
    done
    wait
    echo "[batch$batch_idx] elapsed $(( $(date +%s) - bt ))s"
    batch_idx=$((batch_idx + 1))
done

echo "===== Phase 2C complete in $(( $(date +%s) - start_t ))s ====="
