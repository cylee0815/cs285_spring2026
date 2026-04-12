# Portfolio IQL

Research-grade offline reinforcement learning framework for 8-asset portfolio
allocation using **Implicit Q-Learning (IQL)**. Built for reproducibility and
extensibility; designed to match the quality bar of ML conference repositories.

## Offline RL Formulation

### MDP Definition

- **State:** Market features (past log returns, rolling volatility, moving averages, momentum signals) and current portfolio weights.
- **Action:** Portfolio weight vector `w ∈ R^8` on the simplex (`Σ w_i = 1, w_i ≥ 0`), parameterized via softmax.
- **Reward:** `r_t = log(1 + w_t^T R_{t+1}) - λ ||w_t - w_{t-1}||_1` (log portfolio return minus turnover penalty).
- **Dataset:** Offline transitions `(s_t, a_t, r_t, s_{t+1})` collected from diverse behavior policies (Dirichlet, equal weight, momentum, risk parity).

### IQL Algorithm

IQL trains three networks:
1. **Q-network** `Q(s, a)` — Bellman update with target V-network
2. **Value network** `V(s)` — expectile regression on Q-values
3. **Policy network** `π(s)` — advantage-weighted behavioral cloning

## Asset Universe

| Ticker | Exposure            |
| :----: | ------------------- |
|  SPY   | US equities         |
|  EEM   | Emerging markets    |
|  TLT   | Long duration bonds |
|  HYG   | High-yield credit   |
|  DBC   | Broad commodities   |
|  GLD   | Gold                |
|  UUP   | US dollar           |
|  SHY   | Cash proxy          |

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
cd final_project2
uv sync
```

## Quick Start — Full Pipeline

Run the entire pipeline end-to-end:

```bash
bash scripts/run_full_pipeline.sh
```

This will:
1. Generate an offline dataset
2. Train the IQL model
3. Run backtest against baselines
4. Generate tearsheet and metrics
5. Aggregate results into summary CSV

## Step-by-Step Usage

### 1. Generate Offline Dataset

```bash
uv run python scripts/build_dataset.py \
    --policy dirichlet \
    --seed 42 \
    --output datasets/dirichlet_dataset.npz
```

Supported behavior policies: `dirichlet`, `equal_weight`, `momentum`, `risk_parity`.

### 2. Train IQL

Using a config file (recommended):

```bash
uv run python scripts/train.py --config configs/iql_default.yaml
```

With CLI overrides:

```bash
uv run python scripts/train.py \
    --config configs/iql_default.yaml \
    --expectile 0.8 \
    --beta 5.0
```

Without config (all CLI args):

```bash
uv run python scripts/train.py \
    --dataset datasets/dirichlet_dataset.npz \
    --steps 100000 \
    --expectile 0.7 \
    --beta 3.0
```

### 3. Backtest

```bash
uv run python scripts/backtest.py \
    --checkpoint checkpoints/iql.pt \
    --dataset datasets/dirichlet_dataset.npz \
    --run_name default
```

Outputs:
- `results/default/metrics.csv` — performance metrics for IQL and baselines
- `results/default/tearsheet.png` — equity curves, drawdown, rolling Sharpe

### 4. Ablation Experiments

Run a hyperparameter sweep:

```bash
uv run python scripts/run_ablation.py --config configs/experiments.yaml
```

Preview the grid without running:

```bash
uv run python scripts/run_ablation.py --config configs/experiments.yaml --dry-run
```

### 5. Aggregate Results

```bash
uv run python analysis/aggregate_results.py --results_dir results
```

Produces `results/summary.csv` with all experiment metrics.

### 6. Regime Analysis

```bash
uv run python analysis/regime_analysis.py \
    --returns results/portfolio_returns.npy \
    --start_date 2008-01-01
```

## Configuration

Configs are YAML files under `configs/`:

- `configs/iql_default.yaml` — default training hyperparameters
- `configs/experiments.yaml` — ablation grid specification

The full typed config system lives in `experiments/configs/default.yaml` with
dataclass-backed loading via `utils/config.py`. Supports dotted CLI overrides:

```bash
uv run python -m training.train_iql \
    --config experiments/configs/default.yaml \
    iql.expectile_tau=0.85 train.seed=7
```

## Evaluation Metrics

| Metric              | Description                        |
| ------------------- | ---------------------------------- |
| Annual Return       | Annualized mean daily return       |
| Annual Volatility   | Annualized standard deviation      |
| Sharpe Ratio        | Risk-adjusted return (ann.)        |
| Max Drawdown        | Largest peak-to-trough decline     |
| Cumulative Return   | Total return over test period      |
| Turnover            | Mean L1 distance between weights   |

### Baselines

- **Equal Weight (1/N)** — constant equal allocation
- **Momentum** — softmax of trailing mean returns
- **Risk Parity** — inverse-volatility weighting
- **Buy and Hold** — equal weight at start, drift with returns

## Directory Structure

```
final_project2/
├── algorithms/       # IQL agent
├── analysis/         # results aggregation, regime analysis
├── configs/          # training configs (iql_default.yaml, experiments.yaml)
├── data/             # price download + offline dataset builder
├── env/              # PortfolioEnv simulator
├── evaluation/       # backtest, metrics, baselines
├── experiments/
│   └── configs/      # full typed YAML configs (default.yaml)
├── features/         # feature engineering + normalization
├── models/           # Q / V / policy networks + shared MLP
├── policies/         # behavior policies for dataset generation
├── scripts/          # CLI entrypoints (train, backtest, ablation, pipeline)
├── tests/            # unit + smoke tests
├── training/         # training loop
├── utils/            # config loader, seeding, logging, replay buffer
├── checkpoints/      # saved model weights (gitignored)
├── datasets/         # generated .npz files (gitignored)
└── results/          # experiment outputs (gitignored)
```

## Running Tests

```bash
uv run pytest
```

## Reproducibility

- All experiments use fixed random seeds
- Config files are saved alongside checkpoints
- Deterministic PyTorch settings enabled by default
- Walk-forward validation prevents data leakage

## Roadmap

- [x] **M0** — Scaffolding, config, seeding, logging
- [x] **M1** — Data ingestion + feature pipeline
- [x] **M2** — Portfolio environment
- [x] **M3** — Offline dataset generation
- [x] **M4** — Q / V / policy network architectures
- [x] **M5** — IQL training loop
- [x] **M6** — Walk-forward backtest + baselines + tear sheet
- [x] **M7** — Research polish + experiment reproducibility

## License

MIT.
