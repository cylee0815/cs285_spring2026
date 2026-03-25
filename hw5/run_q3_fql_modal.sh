#!/bin/bash
# =============================================================================
# Q3: FQL (Section 4.4) — Modal version
# Identical experiment config to run_q3_fql.sh, but jobs are submitted to
# Modal from your local machine instead of running on-cluster.
#
# --detach submits all three blocks immediately and returns; they run in
# parallel on Modal. Monitor progress on the Modal dashboard or via WandB.
#
# Usage (from local machine):
#   cd hw5 && bash run_q3_fql_modal.sh 2>&1 | tee logs/q3_fql_modal.log
#
# Download results after all jobs finish:
#   mkdir -p exp
#   uv run modal volume get hw5-offline-rl-volume / exp
# =============================================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

SEED=285
RUN_GROUP=q3
MODAL_CMD="uv run modal run --detach src/scripts/modal_run.py"

# -----------------------------------------------------------------------------
# [OPTIONAL DEBUG] antmaze-medium — 300K steps
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q3 FQL [OPTIONAL DEBUG]: antmaze-medium alpha sweep {1, 3, 10, 30} ==="
$MODAL_CMD --njobs=4 --training_steps=300000 \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antmaze-medium submitted."

# -----------------------------------------------------------------------------
# [REQUIRED] cube-single — 1M steps
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q3 FQL: cube-single alpha sweep {30, 100, 300, 1000} ==="
$MODAL_CMD --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=30" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=100" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=300" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=1000"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cube-single submitted."

# -----------------------------------------------------------------------------
# [REQUIRED] antsoccer-arena — 1M steps
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q3 FQL: antsoccer-arena alpha sweep {1, 3, 10, 30} ==="
$MODAL_CMD --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antsoccer-arena submitted."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q3 FQL (Modal) all jobs submitted ==="
