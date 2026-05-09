#!/usr/bin/env bash
# Run the online-RL baseline matrix: {ppo, ppo_lstm, sac_dirichlet} × 3 seeds.
#
# Usage:
#   bash scripts/run_online_matrix.sh                    # default budget
#   TOTAL_STEPS=500000 bash scripts/run_online_matrix.sh # longer runs
#   ALGOS="ppo ppo_lstm" SEEDS="0 1" bash scripts/run_online_matrix.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ALGOS=${ALGOS:-"ppo ppo_lstm sac_dirichlet"}
SEEDS=${SEEDS:-"0 1 2"}
TOTAL_STEPS=${TOTAL_STEPS:-200000}
EVAL_INTERVAL=${EVAL_INTERVAL:-20000}
DATASET=${DATASET:-datasets/real_dirichlet.npz}
DEVICE=${DEVICE:-auto}
RESULTS_DIR=${RESULTS_DIR:-results/online}

echo "[matrix] algos='${ALGOS}' seeds='${SEEDS}' steps=${TOTAL_STEPS} device=${DEVICE}"

for algo in ${ALGOS}; do
  for seed in ${SEEDS}; do
    run_name="${algo}_seed${seed}"
    echo "=== ${run_name} ==="
    uv run python scripts/run_online_baselines.py \
      --algo "${algo}" \
      --seed "${seed}" \
      --dataset "${DATASET}" \
      --total_timesteps "${TOTAL_STEPS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --device "${DEVICE}" \
      --results_dir "${RESULTS_DIR}" \
      --run_name "${run_name}"
  done
done

echo "[matrix] done. Aggregate with: python -c 'import json, glob; ..."
