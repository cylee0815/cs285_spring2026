#!/usr/bin/env bash
# Run FinRL DRLAgent multi-algorithm comparison in the background.
#
# Trains PPO, A2C, DDPG, SAC, TD3 sequentially using FinRL's DRLAgent on the
# same FinRL PortfolioOptimizationEnv with the proposal's chronological split:
#   Train: 2008-2020  |  Val: 2021  |  Test: 2022-2026
#
# Each seed is a separate background process (different WandB runs within the
# same group) so seeds run in parallel.
#
# Usage:
#   ./scripts/run_finrl_compare.sh                         # 3 seeds, default settings
#   ./scripts/run_finrl_compare.sh finrl_mf "" 1995-01-01 "--use_mutual_funds"
#
# Args:
#   $1  run_group (default: finrl_compare)
#   $2  total_timesteps (default: 500000)
#   $3  start_date (default: 2008-01-01; use 1995-01-01 with mutual funds)
#   $4  extra flags (e.g. "--use_mutual_funds --use_macro")

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

RUN_GROUP="${1:-finrl_compare}"
TOTAL_STEPS="${2:-500000}"
START_DATE="${3:-2008-01-01}"
EXTRA="${4:-}"

for SEED in 0 1 2; do
    LOG="logs/finrl_compare_seed${SEED}.log"
    PID_FILE="logs/finrl_compare_seed${SEED}.pid"

    # shellcheck disable=SC2086
    nohup uv run src/scripts/run_finrl_drl.py \
        --run_group="${RUN_GROUP}" \
        --total_timesteps="${TOTAL_STEPS}" \
        --start_date="${START_DATE}" \
        --eval_interval=10000 \
        --n_eval_episodes=10 \
        --save_results \
        --seed="${SEED}" \
        ${EXTRA} \
        > "${LOG}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "finrl_compare  seed=${SEED}  PID=$(cat ${PID_FILE})  log=${LOG}"
done

echo ""
echo "All FinRL-compare runs launched. Monitor with:"
echo "  tail -f logs/finrl_compare_seed0.log"
echo ""
echo "To train only specific algorithms (e.g. PPO + SAC):"
echo "  uv run src/scripts/run_finrl_drl.py --algos ppo sac --run_group=debug --seed=0"
