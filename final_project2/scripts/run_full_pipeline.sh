#!/usr/bin/env bash
# =============================================================================
# Full reproducible pipeline for Portfolio IQL.
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Steps:
#   1. Generate offline dataset
#   2. Train IQL model
#   3. Run backtest + generate tearsheet
#   4. Aggregate results
#
# Requirements:
#   - uv sync (dependencies installed)
#   - Working directory should be the project root
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo "Portfolio IQL — Full Pipeline"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

# -----------------------------------------------
# 1. Generate offline dataset
# -----------------------------------------------
echo ""
echo "[Step 1/5] Generating offline dataset..."
mkdir -p datasets

uv run python scripts/build_dataset.py \
    --policy dirichlet \
    --seed 42 \
    --n-steps 500 \
    --n-assets 8 \
    --output datasets/dirichlet_dataset.npz

echo "Dataset generated: datasets/dirichlet_dataset.npz"

# -----------------------------------------------
# 2. Train IQL model
# -----------------------------------------------
echo ""
echo "[Step 2/5] Training IQL model..."

uv run python scripts/train.py \
    --config configs/iql_default.yaml \
    --checkpoint_dir checkpoints

echo "Training complete."

# -----------------------------------------------
# 3. Run backtest + generate tearsheet
# -----------------------------------------------
echo ""
echo "[Step 3/5] Running backtest..."

uv run python scripts/backtest.py \
    --checkpoint checkpoints/iql.pt \
    --dataset datasets/dirichlet_dataset.npz \
    --run_name default

echo "Backtest complete."

# -----------------------------------------------
# 4. Aggregate results
# -----------------------------------------------
echo ""
echo "[Step 4/5] Aggregating results..."

uv run python analysis/aggregate_results.py \
    --results_dir results \
    --output results/summary.csv

echo "Results aggregated."

# -----------------------------------------------
# 5. Summary
# -----------------------------------------------
echo ""
echo "[Step 5/5] Pipeline complete!"
echo "============================================"
echo "Outputs:"
echo "  Dataset:    datasets/dirichlet_dataset.npz"
echo "  Checkpoint: checkpoints/iql.pt"
echo "  Metrics:    results/default/metrics.csv"
echo "  Tearsheet:  results/default/tearsheet.png"
echo "  Summary:    results/summary.csv"
echo "============================================"
