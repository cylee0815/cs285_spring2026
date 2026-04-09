# Experimental Plan: O2O Deep RL for Portfolio Optimization

## Environments

### Custom PortfolioEnv (`src/envs/portfolio_env.py`)
- Gymnasium env with random episode start times
- Observation: `[weights(8), per_asset_TA(8x6=48), macro(8?), sentiment(1+384?)]`
- Supports offline dataset construction (random-start episodes) and online training
- `ActionBoundedWrapper` (`action_bounded_wrapper.py`) available for algorithms that don't natively output simplex actions

### FinRLPortfolioWrapper (`src/envs/finrl_wrapper.py`)
- Adapts FinRL's old-gym env to Gymnasium; richer observations (MACD, RSI, CCI, turbulence via configurable `time_window`)
- Converts Dirichlet weights via `log(w)` trick so `softmax(log(w)) = w`
- Used optionally for online phase only (can't do random-start episodes needed for offline)

---

## Reward

- `log_return` — raw portfolio return
- `diff_sharpe` — differential Sharpe ratio (dense risk-adjusted signal)

Both reward types work with both custom and FinRL environments.

---

## Data Sources

### Price Data
`data_utils.py` via `yfinance`

### Macro Features (`macro_features.py`)
8 FRED series: DGS10, DFF, yield_spread, CPI_YoY, UNRATE, GDP_growth, UMich_sentiment, NFCI. Requires `FRED_API_KEY`; falls back to zeros.

### Sentiment Features (`sentiment_features.py`)
SF Fed Daily News Sentiment Index + Alpaca News 384-d embeddings (requires `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`).

---

## Ticker Universes (8-asset "All-Weather" macro portfolio)

### Standard ETFs (Orthogonal 8)
```
[SPY, EEM, TLT, HYG, DBC, GLD, UUP, SHY]
```
Data begins 2008 due to UUP/DBC launch dates.

### Mutual Fund Proxies
```
[VFINX, VEIEX, VUSTX, VWEHX, PCRIX, USERX, VMFXX, VFISX]
```
Triggered via `--use_mutual_funds`. Data back to 1990s.

---

## Chronological Splits

| Split | Period | Purpose |
|-------|--------|---------|
| **Train (Offline)** | 2008-01-01 to 2020-12-31 | Offline dataset (GFC 2008 + COVID 2020) |
| **Validation** | 2021-01-01 to 2021-12-31 | Hyperparameter tuning |
| **Test (O2O)** | 2022-01-01 to 2026-03-31 | Online fine-tuning + evaluation |

The offline agent overfits to the historical stocks-down/bonds-up correlation. In 2022, rapid inflation broke this 40-year correlation — both stocks and bonds crashed simultaneously. This distribution shift is the key motivation for O2O adaptation.

---

## Classical Portfolio Benchmarks

Run via `src/scripts/run_classical.py`.

| Strategy | Method | Notes |
|----------|--------|-------|
| **Equal Weight (1/N)** | `w_i = 1/N` | Static allocation |
| **Inverse Volatility** | `w_i ∝ 1/σ_i` | Uses rolling_std from obs features |
| **Markowitz MVO** | `argmax(μᵀw - λ/2 wᵀΣw)` | Fit on train split returns |

---

## Online Learning

### Custom Implementations (`run.py`)

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **PPO** | `ppo_config` | Feedforward MLP policy |
| **PPO-LSTM** | `ppo_lstm_config` | Recurrent policy |
| **PPO-Transformer** | `ppo_transformer_config` | Attention-based policy |

### Novel Online Algorithm

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **SAC-Dirichlet** | `sac_dirichlet_config` | Dirichlet policy on simplex, exact entropy |

### Stable-Baselines3 Baselines (`run_sb3.py`)

| Algorithm | Config |
|-----------|--------|
| **PPO (SB3)** | `ppo_sb3_config` |
| **A2C (SB3)** | `a2c_config` |
| **SAC (SB3)** | `sac_sb3_config` |
| **DDPG (SB3)** | `ddpg_config` |
| **TD3 (SB3)** | `td3_config` |
| **TQC (SB3)** | `tqc_config` |

### FinRL Native Baselines (`run_finrl_drl.py`)
FinRL's built-in A2C, PPO, DDPG, TD3, SAC on FinRL env.

---

## Offline Learning

All run via `src/scripts/run_offline.py --base_config=<algo>`.

### Behavior-Based Baselines

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **BC** | `bc_config` | Supervised imitation: `max E[log π(a\|s)]` via Dirichlet NLL |
| **Fisher-BC** | `fisher_bc_config` | BC with Fisher-Rao distance loss on simplex |

### Actor-Regularized Offline RL

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **TD3+BC** | `td3_bc_config` | TD3 + BC regularization: `L = -λQ(s,π(s)) + \|\|π(s)-a\|\|²` |
| **AWAC** | `awac_config` | Advantage-weighted actor-critic: `π ∝ exp(A/β) * π_behavior` |

### Conservative Offline RL

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **CQL (vanilla)** | `cql_vanilla_config` | Standard conservative penalty (ablation of Geodesic-CQL) |
| **Geodesic-CQL** | `cql_geodesic_config` | Fisher-Rao geodesic distance penalty (novel) |

### Proposed Addition: Multi-Step Returns

To investigate whether multi-step targets improve sample efficiency in offline learning:

**Implementation needed:**
1. In `replay_buffer.py`: add an n-step return buffer that accumulates `n` transitions and computes the discounted return before storing
2. In `cql_geodesic.py`: modify the Bellman target to use the n-step target state and cumulative discounted reward: `r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n Q(s_{t+n}, a_{t+n})`
3. In `cql_geodesic_config.py`: add `n_step: int = 1`
4. In `run_o2o.py`: wire `--n_step` through the argparser

**Test n ∈ {1, 3, 5, 10}**. Key hypothesis: multi-step targets reduce variance of long-horizon credit assignment (good for noisy financial rewards), but may exacerbate offline distribution shift since the n-step target relies on the behavior policy for n steps. This tension with CQL's conservatism is itself an interesting finding to report.

### Advantage-Based Offline RL

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **IQL** | `iql_config` | Implicit Q-Learning with expectile regression |

### Ensemble Offline RL

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **EDAC** | `edac_config` | Ensemble Diversified Actor-Critic (N=10 Q-networks) |

### Generative Policy Constraint

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **BCQ** | `bcq_config` | Batch-Constrained Q-Learning with VAE-generated candidates |

---

## Model-Based RL

Run via `src/scripts/run_offline.py --base_config=<algo>`.

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **MBPO** | `mbpo_config` | Ensemble dynamics → short synthetic rollouts → SAC |
| **MOPO** | `mopo_config` | MBPO + uncertainty penalty: `r' = r - λ * σ` |

Ensemble dynamics model: 7 probabilistic MLPs, top 5 elites. Rollout length = 1 (portfolio dynamics are highly non-stationary).

---

## Sequence Modeling

Run via `src/scripts/run_offline.py --base_config=<algo>`.

| Algorithm | Config | Notes |
|-----------|--------|-------|
| **Decision Transformer** | `decision_transformer_config` | `π(a_t \| s_t, RTG_t)` via causal transformer |
| **Trajectory Transformer** | `trajectory_transformer_config` | Discretized (obs, action) autoregressive model |

---

## Offline-to-Online (O2O) Pipeline

Run via `src/scripts/run_o2o.py --phase=o2o`.

### Novel Algorithms (CS 285 Project Contributions)

**1. SAC-Dirichlet** (`src/agents/sac_dirichlet.py`, `src/networks/dirichlet_policy.py`)
- Replaces Gaussian-softmax policy with true Dirichlet distribution on portfolio simplex
- `α = softplus(f_θ(s)) + 1` ensures α > 1 (unimodal)
- Exact entropy computation; reparameterized sampling via Gamma trick

**2. Geodesic-CQL** (`src/agents/cql_geodesic.py`)
- Fisher-Rao geodesic distance as CQL penalty metric:
  `d_FR(w1, w2) = 2 * arccos(Σ √(w_i * w'_i))`
- Natural Riemannian metric on the probability simplex

**3. Regime-Conditioned POMDP** (`src/networks/regime_encoder.py`)
- GRU-based `RegimeEncoder` → regime belief state `h_t`
- Actor and critic conditioned on `h_t`: `π(a|s,h)`, `Q(s,a,h)`

**4. O2O Adaptive Pipeline** (`src/agents/o2o_agent.py`)
- Offline pre-training (Geodesic-CQL) → online fine-tuning (SAC-Dirichlet)
- Regime KL divergence adaptively scales CQL weight: `sigmoid(λ * KL(h_offline || h_online))`

### O2O Pipeline Phases

| Phase | Algorithm | Data |
|-------|-----------|------|
| Offline pre-training | Geodesic-CQL | 2008-2020 historical trajectories |
| Online fine-tuning | SAC-Dirichlet | 2022-2026 test period |
| Adaptive bridge | O2O Agent | KL-based CQL weight scheduling |

---

## Multi-Step Return Experiments

Test `n ∈ {1, 3, 5, 10}` for offline algorithms via `--n_step`:
```
r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n Q(s_{t+n}, a_{t+n})
```
Key trade-off: better credit assignment vs. amplified distribution shift.

---

## Ablation Studies

### Input Modalities
Test combinations: price features only → +macro → +sentiment → +news embeddings

### Reward Functions
Compare: `log_return` vs `diff_sharpe`

### Environment
Evaluate both: Custom PortfolioEnv vs FinRL Environment

### Architecture Ablation
Key algorithms (IQL, TD3+BC) tested with both MLP+softmax and Dirichlet policy.

---

## Complete Experiment Commands

### Classical Benchmarks
```bash
uv run src/scripts/run_classical.py --run_group=baselines --seed=0
uv run src/scripts/run_classical.py --run_group=baselines --seed=1
uv run src/scripts/run_classical.py --run_group=baselines --seed=2
```

### Online Baselines (Custom Implementations)
```bash
uv run src/scripts/run.py --base_config=ppo --use_finrl --reward_type=diff_sharpe --seed=0
uv run src/scripts/run.py --base_config=ppo_lstm --use_finrl --reward_type=diff_sharpe --seed=0
uv run src/scripts/run.py --base_config=ppo_transformer --use_finrl --reward_type=diff_sharpe --seed=0
```

### Online Baselines (SB3)
```bash
uv run src/scripts/run_sb3.py --base_config=a2c --use_finrl --seed=0
uv run src/scripts/run_sb3.py --base_config=ppo_sb3 --use_finrl --seed=0
uv run src/scripts/run_sb3.py --base_config=sac_sb3 --use_finrl --seed=0
uv run src/scripts/run_sb3.py --base_config=ddpg --use_finrl --seed=0
uv run src/scripts/run_sb3.py --base_config=td3 --use_finrl --seed=0
uv run src/scripts/run_sb3.py --base_config=tqc --use_finrl --seed=0
```

### Online Baselines (FinRL Native)
```bash
uv run src/scripts/run_finrl_drl.py --seed=0
```

### SAC-Dirichlet Online Baseline
```bash
uv run src/scripts/run_o2o.py --phase=sac --reward_type=diff_sharpe --seed=0
```

### Offline Baselines (Behavior-Based)
```bash
uv run src/scripts/run_offline.py --base_config=bc --seed=0
uv run src/scripts/run_offline.py --base_config=fisher_bc --seed=0
```

### Offline Baselines (Actor-Regularized)
```bash
uv run src/scripts/run_offline.py --base_config=td3_bc --seed=0
uv run src/scripts/run_offline.py --base_config=awac --seed=0
```

### Offline Baselines (Conservative)
```bash
uv run src/scripts/run_offline.py --base_config=cql_vanilla --seed=0
uv run src/scripts/run_o2o.py --phase=offline --seed=0  # Geodesic-CQL
```

### Offline Baselines (Other)
```bash
uv run src/scripts/run_offline.py --base_config=iql --seed=0
uv run src/scripts/run_offline.py --base_config=edac --seed=0
uv run src/scripts/run_offline.py --base_config=bcq --seed=0
```

### Model-Based Offline RL
```bash
uv run src/scripts/run_offline.py --base_config=mbpo --seed=0
uv run src/scripts/run_offline.py --base_config=mopo --seed=0
```

### Sequence Modeling
```bash
uv run src/scripts/run_offline.py --base_config=decision_transformer --seed=0
uv run src/scripts/run_offline.py --base_config=trajectory_transformer --seed=0
```

### Multi-Step Returns (Geodesic-CQL)
```bash
uv run src/scripts/run_o2o.py --phase=offline --n_step=3 --seed=0
uv run src/scripts/run_o2o.py --phase=offline --n_step=5 --seed=0
uv run src/scripts/run_o2o.py --phase=offline --n_step=10 --seed=0
```

### O2O Pipeline
```bash
uv run src/scripts/run_o2o.py --phase=o2o --seed=0
uv run src/scripts/run_o2o.py --phase=o2o --use_finrl_online --seed=0
uv run src/scripts/run_o2o.py --phase=o2o --use_mutual_funds --start_date=1995-01-01 --seed=0
```

### Ablations: Modality
```bash
uv run src/scripts/run_o2o.py --phase=o2o --use_macro --seed=0
uv run src/scripts/run_o2o.py --phase=o2o --use_macro --use_sentiment --seed=0
uv run src/scripts/run_o2o.py --phase=o2o --use_macro --use_sentiment --use_alpaca_embeddings --seed=0
```

### Ablations: Reward
```bash
uv run src/scripts/run_o2o.py --phase=o2o --reward_type=log_return --seed=0
uv run src/scripts/run_o2o.py --phase=o2o --reward_type=diff_sharpe --seed=0
### Ablations: Custom env (for comparison with FinRL results)
```bash
uv run src/scripts/run.py --base_config=ppo --reward_type=diff_sharpe --seed=0
uv run src/scripts/run.py --base_config=ppo_lstm --reward_type=diff_sharpe --seed=0
uv run src/scripts/run.py --base_config=ppo_transformer --reward_type=diff_sharpe --seed=0
```

---

## Logging

All experiments logged to Weights & Biases.

Metrics: cumulative return, Sharpe ratio, max drawdown, turnover, portfolio entropy.

---

## Expected Research Contributions

1. Geometry-aware offline RL using **Fisher-Rao distance on the simplex**
2. **Dirichlet policy SAC** for constrained portfolio weights
3. **Offline-to-online adaptation under regime shift** with KL-based conservatism
4. Integration of **macro + sentiment features** in RL portfolio optimization
5. Comprehensive comparison of **15+ offline/online algorithms** on a realistic portfolio task
