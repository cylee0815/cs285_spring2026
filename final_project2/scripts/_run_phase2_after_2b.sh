#!/usr/bin/env bash
# Master orchestrator that runs the post-Phase-2B phases sequentially.
# Launched once Phase 2B's background task notifies completion.
#
# Schedule:
#   Phase 2A (24 offline runs, 4-way parallel) — ~90 min
#   Phase 2C (6 O2O runs, 2-way parallel) || Phase 2D (3 GRPO ablation runs, 3-way parallel)
#   Aggregate + plot — ~1 min
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJ_ROOT"

LOG_DIR="$PROJ_ROOT/logs"
mkdir -p "$LOG_DIR"

start_t=$(date +%s)

echo "===== Phase 2A start ====="
bash scripts/_run_phase2a.sh > "$LOG_DIR/phase2a_MASTER.log" 2>&1
echo "[phase2a] elapsed $(( $(date +%s) - start_t ))s"

# Aggregate Phase 2A early so Phase 2C can pick up best-checkpoint references
# (currently aggregator just unions per-run.csvs; Phase 2C doesn't depend on
# this output, but it's cheap and gives an intermediate artifact).
uv run python scripts/aggregate_phase2.py > "$LOG_DIR/aggregate_post2a.log" 2>&1 || true

echo "===== Phase 2C + 2D in parallel ====="
b2=$(date +%s)
bash scripts/_run_phase2c.sh > "$LOG_DIR/phase2c_MASTER.log" 2>&1 &
P2C_PID=$!
bash scripts/_run_phase2d_grpo_ablation.sh > "$LOG_DIR/phase2d_MASTER.log" 2>&1 &
P2D_PID=$!
wait $P2C_PID $P2D_PID
echo "[phase2c+2d] elapsed $(( $(date +%s) - b2 ))s"

echo "===== Aggregate + plot ====="
uv run python scripts/aggregate_phase2.py > "$LOG_DIR/aggregate_final.log" 2>&1
uv run python scripts/plot_phase2.py >> "$LOG_DIR/aggregate_final.log" 2>&1

echo "===== All post-2B phases complete in $(( $(date +%s) - start_t ))s ====="
