#!/bin/bash
# =============================================================================
# Q2: IQL (Section 3.2)
# Task order: antmaze-medium (debug, 300K steps) → cube-single → antsoccer-arena
#
# Usage (interactive GPU session):
#   cd hw5 && bash run_q2_iql.sh 2>&1 | tee logs/q2_iql.log
#
# Usage (SLURM batch — uncomment #SBATCH lines, then: sbatch run_q2_iql.sh):
##SBATCH --job-name=hw5_q2_iql
##SBATCH --gres=gpu:1
##SBATCH --time=08:00:00
##SBATCH --cpus-per-task=8
##SBATCH --mem=32G
##SBATCH --output=logs/slurm_q2_iql_%j.log
#
# Expected runtime (4 parallel jobs/task, 1M steps unless noted):
#   V100: antmaze ~0.5h | cube-single ~3-5h | antsoccer ~3-5h
#         → antmaze + cube-single fits well; antsoccer may need a second job
#   A100: antmaze ~0.3h | cube-single ~2-3h | antsoccer ~2-4h
#         → all three fit comfortably in 8h
#
# IQL alpha note: recommended start is {1, 3, 10}; sweeping {1, 3, 10, 30}
# fills all 4 parallel slots and satisfies the "at least 3 values" report requirement.
#
# Deliverables produced:
#   exp/q2/sd285_*_iql_cube-single*_a<alpha>/    (best -> rename to cube-single/)
#   exp/q2/sd285_*_iql_antsoccer*_a<alpha>/      (best -> rename to antsoccer-arena/)
# =============================================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

SEED=285
RUN_GROUP=q2

# -----------------------------------------------------------------------------
# [OPTIONAL DEBUG] antmaze-medium — 300K steps, target >60% at 200K steps.
# Runs first as a quick sanity check. Comment out this block to skip.
# IQL tip: with correct implementation and good alpha, success should appear
# around 100-150K steps on antmaze-medium.
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q2 IQL [OPTIONAL DEBUG]: antmaze-medium alpha sweep {1, 3, 10, 30} ==="
uv run src/scripts/run.py --njobs=4 --training_steps=300000 \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antmaze-medium done."

# -----------------------------------------------------------------------------
# [REQUIRED] cube-single — 1M steps, target >60% success rate.
# Alpha sweep: {1, 3, 10, 30}.
# Report must include training curves for ≥3 different alpha values on this task.
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q2 IQL: cube-single alpha sweep {1, 3, 10, 30} ==="
uv run src/scripts/run.py --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cube-single done."

# -----------------------------------------------------------------------------
# [REQUIRED] antsoccer-arena — 1M steps, target >5% success rate.
# Alpha sweep: {1, 3, 10, 30}.
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q2 IQL: antsoccer-arena alpha sweep {1, 3, 10, 30} ==="
uv run src/scripts/run.py --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antsoccer-arena done."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q2 IQL complete ==="
