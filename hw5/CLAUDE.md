# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CS285 HW5: Offline Reinforcement Learning. Implements three offline RL algorithms using PyTorch and OGBench environments:
- **Q1**: SAC+BC (Soft Actor-Critic with Behavior Cloning)
- **Q2**: IQL (Implicit Q-Learning)
- **Q3**: FQL (Flow Q-Learning)

## Commands

### Running Experiments

**Local (single GPU):**
```bash
uv run src/scripts/run.py --run_group=q1 --base_config=sacbc \
  --env_name=cube-single-play-singletask-task1-v0 --seed=0
```

**Local (4 parallel jobs on one GPU):**
```bash
uv run src/scripts/run.py --njobs=4 \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=285 --alpha=30" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=285 --alpha=100" \
  ...
```

**On Modal (cloud GPU):**
```bash
uv run modal run src/scripts/modal_run.py --run_group=q1 --base_config=sacbc \
  --env_name=cube-single-play-singletask-task1-v0 --seed=0
# Use --detach to run in background
```

**Convenience scripts** (run from `hw5/` directory):
```bash
bash run_q1_sacbc.sh 2>&1 | tee logs/q1_sacbc.log
bash run_q2_iql.sh   2>&1 | tee logs/q2_iql.log
bash run_q3_fql.sh   2>&1 | tee logs/q3_fql.log
```

**SLURM**: Each `.sh` script has commented-out `#SBATCH` headers — uncomment and use `sbatch`.

**Download logs from Modal:**
```bash
mkdir -p exp && uv run modal volume get hw5-offline-rl-volume / exp
```

**Prefetch datasets:**
```bash
bash prefetch_datasets.sh
```

### Key CLI Arguments

| Argument | Description |
|---|---|
| `--base_config` | `sacbc`, `iql`, or `fql` |
| `--env_name` | OGBench environment name |
| `--seed` | Random seed (assignment uses 285) |
| `--alpha` | BC coefficient (SAC+BC, FQL) or advantage scale (IQL) |
| `--expectile` | Expectile for IQL value regression (default 0.9) |
| `--training_steps` | Total gradient steps (default 1M) |
| `--njobs` | Parallel job count for time-sliced GPU sharing |

## Architecture

### Directory Structure
```
src/
├── agents/          # Algorithm implementations
├── configs/         # Hyperparameter configs per algorithm
├── infrastructure/  # Replay buffer, logging, distributions, pytorch utils
├── networks/        # Neural network building blocks
└── scripts/         # Entry points: run.py (local), modal_run.py (cloud)
exp/                 # Output logs (train.csv, eval.csv, flags.json)
```

### Agent Implementations (`src/agents/`)

**SACBCAgent** (`sacbc_agent.py`): Actor-critic with behavior cloning regularization.
- `update_q()`: TD learning; target uses **average** (not min) of ensemble Q-values
- `update_actor()`: Maximizes Q + BC MSE loss on dataset actions + entropy
- `update_beta()`: Dual gradient descent; target entropy = `-action_dim / 2`
- Evaluation: mode of tanh-transformed Gaussian

**IQLAgent** (`iql_agent.py`): Separates policy optimization from Q-learning via a value network.
- `update_v()`: Expectile regression on `(Q - V)`; asymmetric weight `(1-τ)` below, `τ` above
- `update_q()`: TD bootstrapped by `V(s')` (not `max_a Q`)
- `update_actor()`: Advantage-weighted BC: `exp(α * (Q(s,a) - V(s))) * log π(a|s)`
- Evaluation: mode of distribution, clamped to `[-1, 1]`

**FQLAgent** (`fql_agent.py`): Flow-based policy with distillation.
- Two actors: `bc_actor` (flow/diffusion policy) and `onestep_actor` (distilled, used for eval and Bellman backup)
- `update_bc_actor()`: Regresses on diffusion path: `a_t = (1-t)*z + t*a`
- `update_onestep_actor()`: Distillation loss (`alpha` weight) + Q-maximization loss
- `update_q()`: Uses `onestep_actor` for next-state actions; uses **average** of ensemble Q-values
- Critical: clip actions to `[-1, 1]` when feeding to critic, but **not** in distillation loss

### Networks (`src/networks/rl_networks.py`)

- `Policy`: Gaussian with tanh transform; state-dependent or fixed std
- `VectorFieldPolicy`: Takes `(obs, action, time)` → velocity vector for flow models
- `EnsembleCritic`: Two independent Q-networks; returns `(2, batch, 1)` tensor
- `Value`: Single V(s) network
- `build_mlp()` / `build_ensemble_mlp()` in `infrastructure/pytorch_util.py`

### Training Loop (`src/scripts/run.py`)

1. Load OGBench dataset and environment
2. For 1M steps: sample batch → `agent.update()` → log metrics every 10K steps
3. Evaluate every 100K steps: 25 rollouts, measure `info["success"]`
4. Logs to `exp/{run_group}/sd{seed}_*_{algo}_{env}_a{alpha}/` as `train.csv` and `eval.csv`

### Configs (`src/configs/`)

Each config file exports a factory function (`sacbc_config`, `iql_config`, `fql_config`) returning a dict with `agent_kwargs`, optimizer factories, and the dataset loader. Configs use `ml_collections` for structured hyperparameters.

## Environments and Success Targets

| Environment | Steps | Algorithm Targets |
|---|---|---|
| `antmaze-medium-navigate-singletask-task1-v0` | 200K (debug) | SAC+BC >80%, IQL >60%, FQL >80% |
| `cube-single-play-singletask-task1-v0` | 1M | SAC+BC >75%, IQL >60%, FQL >80% |
| `antsoccer-arena-navigate-singletask-task1-v0` | 1M | SAC+BC >5%, IQL >5%, FQL >30% |

Alpha sweep conventions:
- **cube-single**: `{30, 100, 300, 1000}`
- **antsoccer / antmaze**: `{1, 3, 10, 30}`

## Key Implementation Notes

- All learning is from a fixed offline dataset — **no environment interaction during training**
- SAC+BC and FQL use **average** of critic ensemble for targets; IQL uses **min**
- FQL's `bc_actor` is only used for distillation, never for evaluation or Bellman backup
- BFloat16 autocast (`infrastructure/pytorch_util.py:autocast()`) is used on CUDA
- WandB project: `cs285_hw5` (requires `wandb login`)
