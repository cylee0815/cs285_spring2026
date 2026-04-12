## Project Overview

This repository implements a **research-grade offline reinforcement learning framework for portfolio allocation** using **Implicit Q-Learning (IQL)**.

The goal is to build a **clean, reproducible, extensible research codebase** comparable to modern machine learning research repositories used in top conferences (e.g., NeurIPS, ICML, ICLR).

The system should allow experimentation with offline RL algorithms for portfolio optimization using historical financial data.

The codebase must prioritize:

* Reproducibility
* Modular design
* Experiment tracking
* Clear separation between data, environment, algorithms, and evaluation
* Extensibility for future research

---

# Core Problem

We train an RL agent to allocate capital across **8 assets: [SPY, EEM, TLT, HYG, DBC, GLD, UUP, SHY]**.

At each timestep the agent outputs portfolio weights:

w ∈ ℝ⁸

subject to:

Σ w_i = 1
w_i ≥ 0

The objective is to maximize long-term risk-adjusted returns.

The system must support **offline reinforcement learning using historical market data**.

---

# MDP Definition

## State

State includes market features and current portfolio allocation.

Example components:

Market features:

* past k log returns
* rolling volatility
* moving averages
* momentum signals

Portfolio state:

* current portfolio weights
* previous portfolio weights (for turnover penalty)

Example state vector:

s_t = [
r_{t-20:t},
volatility_{t-20:t},
moving_averages,
momentum_features,
current_portfolio_weights
]

All features must be normalized.

---

## Action

Action is an 8-dimensional weight vector.

Constraint:

Σ w_i = 1

Implementation requirement:

Use **softmax parameterization** for the policy network.

Future extensions should allow:

* long-short portfolios
* leverage constraints
* position limits

---

## Reward

Primary reward:

r_t = log(1 + w_t^T R_{t+1})

Where:

R_{t+1} = (P_{t+1} − P_t) / P_t

Transaction costs must be modeled:

r_t = log(1 + w_t^T R_{t+1}) − λ ||w_t − w_{t−1}||₁

Configurable parameter:

transaction_cost_lambda

Future extension:

Differential Sharpe ratio reward.

---

# Offline RL Dataset

Offline RL requires a dataset:

(s_t, a_t, r_t, s_{t+1})

The repository must implement **behavior policy generators** to create diverse action distributions.

Behavior policies should include:

1. Random portfolios sampled from Dirichlet distribution
2. Equal weight portfolio
3. Momentum portfolio
4. Risk parity portfolio

The dataset builder must:

* simulate trajectories
* compute rewards
* store transitions efficiently
* support large datasets

Preferred storage format:

Parquet or HDF5.

---

# Algorithm: Implicit Q-Learning

Implement **Implicit Q-Learning (IQL)**.

Three neural networks:

### Q Network

Input:
(state, action)

Output:
Q-value

Architecture:

MLP
256
256
1

---

### Value Network

Input:
state

Output:
V(s)

Architecture:

MLP
256
256
1

---

### Policy Network

Input:
state

Output:
portfolio weights

Architecture:

MLP
256
256
8
softmax

---

# IQL Training Procedure

Training must follow standard IQL:

Step 1: Value network via expectile regression

V(s) ≈ expectile_τ(Q(s,a))

τ ∈ [0.7, 0.9]

---

Step 2: Q network Bellman update

Q(s,a) = r + γ V(s')

Use target networks.

---

Step 3: Policy improvement

Advantage:

A(s,a) = Q(s,a) − V(s)

Weight actions using:

exp(β A(s,a))

Policy trained via advantage-weighted behavioral cloning.

---

# Training Requirements

The training loop must support:

* gradient clipping
* target networks
* reward normalization
* configurable hyperparameters

Important hyperparameters:

discount γ = 0.99
expectile τ
advantage temperature β
learning rates

---

# Evaluation Protocol

Evaluation must follow **financial backtesting best practices**.

Implement **walk-forward validation**:

Example:

train: 2005–2016
validation: 2016–2018
test: 2018–2023

Metrics:

* annual return
* Sharpe ratio
* maximum drawdown
* turnover
* volatility

---

# Baselines

The system must implement classical portfolio baselines:

Equal weight
Mean-variance optimization
Risk parity
Momentum portfolio

These baselines are required for fair comparison.

---

# Experiment Management

Experiments must be reproducible.

Requirements:

* configuration files
* deterministic seeds
* experiment logging
* versioned results

Use YAML configuration files.

Recommended tools:

Hydra or similar config framework.

Log:

* training metrics
* evaluation metrics
* hyperparameters
* checkpoints

---

# Reproducibility

Every experiment must be reproducible.

Requirements:

* fixed random seeds
* configuration snapshots
* dataset versioning
* deterministic PyTorch settings when possible

---

# Project Structure

project/

data/
download_data.py
dataset_builder.py

env/
portfolio_env.py

features/
feature_engineering.py
normalization.py

models/
mlp.py
q_network.py
value_network.py
policy_network.py

algorithms/
iql.py

training/
train_iql.py

evaluation/
backtest.py
metrics.py
baselines.py

experiments/
configs/

utils/
logging.py
seed.py

tests/

---

# Evaluation Output

Backtests should produce:

* equity curve
* drawdown curve
* portfolio weights over time

Save results in:

results/

---

# Code Quality

All code must include:

* docstrings
* type hints
* modular functions
* unit tests for core components

Critical components that must be tested:

* environment transitions
* reward computation
* dataset generation

---

# Future Research Extensions

Design the system so it can easily support:

1. Differential Sharpe reward
2. Regime detection
3. Online fine-tuning
4. Alternative RL algorithms
5. Long-short portfolios
6. Risk-aware RL objectives

---

# Implementation Philosophy

The code should resemble **modern research repositories used in ML conferences**.

Priorities:

1. clarity
2. reproducibility
3. extensibility
4. correctness

Avoid monolithic scripts.

Prefer modular design.

Ensure components can be reused for future RL research.