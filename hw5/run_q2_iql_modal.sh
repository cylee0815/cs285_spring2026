#!/bin/bash
# =============================================================================
# Q2: IQL (Section 3.2) — Modal version
# Identical experiment config to run_q2_iql.sh, but jobs are submitted to
# Modal from your local machine instead of running on-cluster.
#
# Each block submits ONE Modal container (DEFAULT_GPU=T4) that runs 4 parallel
# worker processes (--njobs=4) via multiprocessing.  Blocks run sequentially
# so you can monitor antmaze first before committing to the longer runs.
#
# Usage (from local machine):
#   cd hw5 && bash run_q2_iql_modal.sh 2>&1 | tee logs/q2_iql_modal.log
#
# Download results after all jobs finish:
#   mkdir -p exp
#   uv run modal volume get hw5-offline-rl-volume / exp
# =============================================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

SEED=285
RUN_GROUP=q2
MODAL_CMD="uv run modal run --detach src/scripts/modal_run.py"

# -----------------------------------------------------------------------------
# [OPTIONAL DEBUG] antmaze-medium — 200K steps
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q2 IQL [OPTIONAL DEBUG]: antmaze-medium alpha sweep {1, 3, 10, 30} ==="
$MODAL_CMD --njobs=4 --training_steps=300000 \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antmaze-medium done."

# -----------------------------------------------------------------------------
# [REQUIRED] cube-single — 1M steps
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q2 IQL: cube-single alpha sweep {30, 100, 300, 1000} ==="
$MODAL_CMD --njobs=4 \
  # "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  # "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  # "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  # "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=30"
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=30" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=100" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=300" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=1000"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cube-single done."

# -----------------------------------------------------------------------------
# [REQUIRED] antsoccer-arena — 1M steps
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q2 IQL: antsoccer-arena alpha sweep {1, 3, 10, 30} ==="
$MODAL_CMD --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antsoccer-arena done."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q2 IQL (Modal) complete ==="
