#!/usr/bin/env bash
# Run TD3 (Twin Delayed DDPG, off-policy) in the background.
# ActionBoundedWrapper applied automatically. Uses Gaussian exploration noise.
# Logs: logs/td3_seed{N}.log

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

RUN_GROUP="${1:-sb3_baselines}"
TOTAL_STEPS="${2:-500000}"

for SEED in 0 1 2; do
    LOG="logs/td3_seed${SEED}.log"
    PID_FILE="logs/td3_seed${SEED}.pid"

    nohup uv run src/scripts/run_sb3.py \
        --run_group="${RUN_GROUP}" \
        --base_config=td3 \
        --total_timesteps="${TOTAL_STEPS}" \
        --eval_interval=10000 \
        --n_eval_episodes=10 \
        --seed="${SEED}" \
        > "${LOG}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "TD3  seed=${SEED}  PID=$(cat ${PID_FILE})  log=${LOG}"
done

echo ""
echo "All TD3 runs launched. Monitor with:"
echo "  tail -f logs/td3_seed0.log"
