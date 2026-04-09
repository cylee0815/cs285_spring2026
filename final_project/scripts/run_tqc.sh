#!/usr/bin/env bash
# Run TQC (Truncated Quantile Critics, distributional SAC) in the background.
# Requires sb3-contrib. If not installed, run: uv sync
# ActionBoundedWrapper applied automatically.
# Logs: logs/tqc_seed{N}.log

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

# Check sb3-contrib is available before launching
if ! uv run python -c "import sb3_contrib" 2>/dev/null; then
    echo "ERROR: sb3-contrib is not installed."
    echo "Run: uv sync"
    echo "(It has been added to pyproject.toml — uv sync will install it.)"
    exit 1
fi

RUN_GROUP="${1:-sb3_baselines}"
TOTAL_STEPS="${2:-500000}"

for SEED in 0 1 2; do
    LOG="logs/tqc_seed${SEED}.log"
    PID_FILE="logs/tqc_seed${SEED}.pid"

    nohup uv run src/scripts/run_sb3.py \
        --run_group="${RUN_GROUP}" \
        --base_config=tqc \
        --total_timesteps="${TOTAL_STEPS}" \
        --eval_interval=10000 \
        --n_eval_episodes=10 \
        --seed="${SEED}" \
        > "${LOG}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "TQC  seed=${SEED}  PID=$(cat ${PID_FILE})  log=${LOG}"
done

echo ""
echo "All TQC runs launched. Monitor with:"
echo "  tail -f logs/tqc_seed0.log"
