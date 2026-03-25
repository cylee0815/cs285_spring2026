#!/bin/bash
# =============================================================================
# Q3: FQL (Section 4.4)
# Task order: antmaze-medium (debug, 300K steps) → cube-single → antsoccer-arena
#
# Usage (interactive GPU session):
#   cd hw5 && bash run_q3_fql.sh 2>&1 | tee logs/q3_fql.log
#
# Usage (SLURM batch — uncomment #SBATCH lines, then: sbatch run_q3_fql.sh):
##SBATCH --job-name=hw5_q3_fql
##SBATCH --gres=gpu:1
##SBATCH --time=08:00:00
##SBATCH --cpus-per-task=8
##SBATCH --mem=32G
##SBATCH --output=logs/slurm_q3_fql_%j.log
#
# Expected runtime (4 parallel jobs/task, 1M steps unless noted):
#   V100: antmaze ~0.5-1h | cube-single ~5-7h | antsoccer ~5-7h
#         → antmaze + cube-single fills 8h; run antsoccer in a separate job
#   A100: antmaze ~0.3h   | cube-single ~3-4h | antsoccer ~4-5h
#         → all three fit in 8h
#
# FQL notes:
#   - one-step actor (pi_omega) is used for evaluation AND Bellman backup
#   - bc_actor (flow policy) is for distillation only; do NOT use it for eval
#   - clip actions to [-1,1] when feeding to the critic, but NOT in distillation loss
#   - use the AVERAGE (not min) of two target Q values in the Bellman backup
#
# Deliverables produced:
#   exp/q3/sd285_*_fql_cube-single*_a<alpha>/    (best -> rename to cube-single/)
#   exp/q3/sd285_*_fql_antsoccer*_a<alpha>/      (best -> rename to antsoccer-arena/)
# =============================================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

SEED=285
RUN_GROUP=q3

# -----------------------------------------------------------------------------
# [OPTIONAL DEBUG] antmaze-medium — 300K steps, target >80% at 200K steps.
# Runs first as a quick sanity check. Comment out this block to skip.
# FQL tip: bc_actor/loss should decrease steadily; onestep_actor/mse should
# follow bc actions closely for large alpha, diverge for small alpha.
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q3 FQL [OPTIONAL DEBUG]: antmaze-medium alpha sweep {1, 3, 10, 30} ==="
uv run src/scripts/run.py --njobs=4 --training_steps=300000 \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antmaze-medium done."

# -----------------------------------------------------------------------------
# [REQUIRED] cube-single — 1M steps, target >80% success rate.
# Alpha sweep: {30, 100, 300, 1000}.
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q3 FQL: cube-single alpha sweep {30, 100, 300, 1000} ==="
uv run src/scripts/run.py --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=30" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=100" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=300" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=${SEED} --alpha=1000"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cube-single done."

# -----------------------------------------------------------------------------
# [REQUIRED] antsoccer-arena — 1M steps, target best success rate >30%.
# Alpha sweep: {1, 3, 10, 30}.
# Note: FQL significantly outperforms SAC+BC and IQL on this harder task.
# -----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q3 FQL: antsoccer-arena alpha sweep {1, 3, 10, 30} ==="
uv run src/scripts/run.py --njobs=4 \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=1" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=3" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=10" \
  "JOB --run_group=${RUN_GROUP} --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=${SEED} --alpha=30"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] antsoccer-arena done."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Q3 FQL complete ==="
