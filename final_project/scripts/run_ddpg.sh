#!/usr/bin/env bash
# Run DDPG (Deep Deterministic Policy Gradient, off-policy) in the background.
# ActionBoundedWrapper applied automatically. Uses Ornstein-Uhlenbeck exploration noise.
# Logs: logs/ddpg_seed{N}.log

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

RUN_GROUP="${1:-sb3_baselines}"
TOTAL_STEPS="${2:-500000}"

for SEED in 0 1 2; do
    LOG="logs/ddpg_seed${SEED}.log"
    PID_FILE="logs/ddpg_seed${SEED}.pid"

    nohup uv run src/scripts/run_sb3.py \
        --run_group="${RUN_GROUP}" \
        --base_config=ddpg \
        --total_timesteps="${TOTAL_STEPS}" \
        --eval_interval=10000 \
        --n_eval_episodes=10 \
        --seed="${SEED}" \
        > "${LOG}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "DDPG  seed=${SEED}  PID=$(cat ${PID_FILE})  log=${LOG}"
done

echo ""
echo "All DDPG runs launched. Monitor with:"
echo "  tail -f logs/ddpg_seed0.log"
