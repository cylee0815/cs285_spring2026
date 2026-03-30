#!/bin/bash
# =============================================================================
# Q1: SAC+BC (Section 2.2)
# Task order: antmaze-medium (debug, 200K steps) → cube-single → antsoccer-arena
#
# Usage (interactive GPU session):
#   cd hw5 && bash run_q1_sacbc.sh 2>&1 | tee logs/q1_sacbc.log
#
# Usage (SLURM batch — uncomment #SBATCH lines, then: sbatch run_q1_sacbc.sh):
##SBATCH --job-name=hw5_q1_sacbc
##SBATCH --gres=gpu:1
##SBATCH --time=08:00:00
##SBATCH --cpus-per-task=8
##SBATCH --mem=32G
##SBATCH --output=logs/slurm_q1_sacbc_%j.log
#
# Expected runtime (4 parallel jobs/task, 1M steps unless noted):
#   V100: antmaze ~0.5-1h | cube-single ~4-6h | antsoccer ~4-6h
#         → antmaze + cube-single fits in 8h; antsoccer may need a second job
#   A100: antmaze ~0.3h   | cube-single ~2-3h | antsoccer ~3-4h
#         → all three fit comfortably in 8h
#
# Deliverables produced:
#   exp/q1/sd285_*_sacbc_cube-single*_a<alpha>/    (best -> rename to cube-single/)
#   exp/q1/sd285_*_sacbc_antsoccer*_a<alpha>/      (best -> rename to antsoccer-arena/)
# =============================================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

SEED=285
RUN_GROUP=q1

# -----------------------------------------------------------------------------
# [OPTIONAL DEBUG] antmaze-medium — 200K steps, target >80% at 100K steps.
# Runs first as a quick sanity check. Comment out this block to skip.
# SAC+BC tip: q_max -> 0, q_min -> -100 when implementation is correct.
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q1 SAC+BC [OPTIONAL DEBUG]: antmaze-medium alpha sweep {1, 3, 10, 30} ==="
uv run src/scripts/run.py --njobs=4 --training_steps=200000 \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antmaze-medium done."

# -----------------------------------------------------------------------------
# [REQUIRED] cube-single — 1M steps, target >75% success rate.
# Alpha sweep: {30, 100, 300, 1000} (tuned assuming mean over action dim in BC term).
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q1 SAC+BC: cube-single alpha sweep {30, 100, 300, 1000} ==="
uv run src/scripts/run.py --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=30" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=100" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=300" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=1000"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cube-single done."

# -----------------------------------------------------------------------------
# [REQUIRED] antsoccer-arena — 1M steps, target >5% success rate.
# Alpha sweep: {1, 3, 10, 30} (smaller BC coefficient needed for harder task).
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q1 SAC+BC: antsoccer-arena alpha sweep {1, 3, 10, 30} ==="
uv run src/scripts/run.py --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antsoccer-arena done."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q1 SAC+BC complete ==="
