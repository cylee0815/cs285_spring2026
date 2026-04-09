#!/usr/bin/env bash
# Run SAC (Soft Actor-Critic, off-policy, max-entropy) in the background.
# ActionBoundedWrapper is applied automatically by run_sb3.py for sac_sb3.
# Logs: logs/sac_sb3_seed{N}.log

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

RUN_GROUP="${1:-sb3_baselines}"
TOTAL_STEPS="${2:-500000}"

for SEED in 0 1 2; do
    LOG="logs/sac_sb3_seed${SEED}.log"
    PID_FILE="logs/sac_sb3_seed${SEED}.pid"

    nohup uv run src/scripts/run_sb3.py \
        --run_group="${RUN_GROUP}" \
        --base_config=sac_sb3 \
        --total_timesteps="${TOTAL_STEPS}" \
        --eval_interval=10000 \
        --n_eval_episodes=10 \
        --seed="${SEED}" \
        > "${LOG}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "SAC  seed=${SEED}  PID=$(cat ${PID_FILE})  log=${LOG}"
done

echo ""
echo "All SAC runs launched. Monitor with:"
echo "  tail -f logs/sac_sb3_seed0.log"
