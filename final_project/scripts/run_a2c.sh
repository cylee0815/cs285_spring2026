#!/usr/bin/env bash
# Run A2C (on-policy, synchronous actor-critic) in the background.
# Logs: logs/a2c_seed{N}.log
# Monitor: tail -f logs/a2c_seed0.log
# Stop:    kill $(cat logs/a2c_seed0.pid)

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from project root

mkdir -p logs

RUN_GROUP="${1:-sb3_baselines}"
TOTAL_STEPS="${2:-500000}"

for SEED in 0 1 2; do
    LOG="logs/a2c_seed${SEED}.log"
    PID_FILE="logs/a2c_seed${SEED}.pid"

    nohup uv run src/scripts/run_sb3.py \
        --run_group="${RUN_GROUP}" \
        --base_config=a2c \
        --total_timesteps="${TOTAL_STEPS}" \
        --eval_interval=10000 \
        --n_eval_episodes=10 \
        --seed="${SEED}" \
        > "${LOG}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "A2C  seed=${SEED}  PID=$(cat ${PID_FILE})  log=${LOG}"
done

echo ""
echo "All A2C runs launched. Monitor with:"
echo "  tail -f logs/a2c_seed0.log"
