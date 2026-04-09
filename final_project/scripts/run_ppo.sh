#!/usr/bin/env bash
# Run the custom PPO implementation (MLP / LSTM / Transformer) in the background.
# This uses run.py, NOT run_sb3.py — it exercises the custom PPOAgent from src/agents/ppo.py.
#
# Usage:
#   ./scripts/run_ppo.sh                           # MLP, 3 seeds
#   ./scripts/run_ppo.sh my_group ppo_lstm         # LSTM variant
#   ./scripts/run_ppo.sh my_group ppo_transformer  # Transformer variant
#
# Args:
#   $1  run_group (default: ppo_custom)
#   $2  base_config: ppo | ppo_lstm | ppo_transformer (default: ppo)
#   $3  total_timesteps (default: 1000000)

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

RUN_GROUP="${1:-ppo_custom}"
CONFIG="${2:-ppo}"
TOTAL_STEPS="${3:-1000000}"

for SEED in 0 1 2; do
    LOG="logs/${CONFIG}_seed${SEED}.log"
    PID_FILE="logs/${CONFIG}_seed${SEED}.pid"

    nohup uv run src/scripts/run.py \
        --run_group="${RUN_GROUP}" \
        --base_config="${CONFIG}" \
        --total_timesteps="${TOTAL_STEPS}" \
        --eval_interval=10000 \
        --n_eval_episodes=10 \
        --seed="${SEED}" \
        > "${LOG}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "${CONFIG}  seed=${SEED}  PID=$(cat ${PID_FILE})  log=${LOG}"
done

echo ""
echo "All ${CONFIG} runs launched. Monitor with:"
echo "  tail -f logs/${CONFIG}_seed0.log"
