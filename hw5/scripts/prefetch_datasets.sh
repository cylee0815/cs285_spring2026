#!/bin/bash
# Pre-download all ogbench datasets used by run_q*.sh scripts.
# Run this once before launching any training job to avoid download failures
# caused by missing ~/.ogbench/data/ directory or cross-device rename issues.
#
# Usage:
#   cd hw5 && bash prefetch_datasets.sh

set -e
cd "$(dirname "$0")"

mkdir -p ~/.ogbench/data

ENVS=(
  "antmaze-medium-navigate-singletask-task1-v0"
  "cube-single-play-singletask-task1-v0"
  "antsoccer-arena-navigate-singletask-task1-v0"
)

for env in "${ENVS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Prefetching dataset for: $env"
  uv run python -c "import ogbench; ogbench.make_env_and_datasets('$env')"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $env"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All datasets prefetched."
