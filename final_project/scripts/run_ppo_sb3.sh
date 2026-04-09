#!/usr/bin/env bash
# Run SB3 PPO in the background (comparison baseline vs. the custom PPO in run.py).
# Logs: logs/ppo_sb3_seed{N}.log

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

RUN_GROUP="${1:-sb3_baselines}"
TOTAL_STEPS="${2:-1000000}"

for SEED in 0 1 2; do
    LOG="logs/ppo_sb3_seed${SEED}.log"
    PID_FILE="logs/ppo_sb3_seed${SEED}.pid"

    nohup uv run src/scripts/run_sb3.py \
        --run_group="${RUN_GROUP}" \
        --base_config=ppo_sb3 \
        --total_timesteps="${TOTAL_STEPS}" \
        --eval_interval=10000 \
        --n_eval_episodes=10 \
        --seed="${SEED}" \
        > "${LOG}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "PPO-SB3  seed=${SEED}  PID=$(cat ${PID_FILE})  log=${LOG}"
done

echo ""
echo "All PPO-SB3 runs launched. Monitor with:"
echo "  tail -f logs/ppo_sb3_seed0.log"
