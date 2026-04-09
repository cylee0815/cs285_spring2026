# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (requires Python 3.10–3.12; use 3.12 for compatibility)
uv sync --python 3.12

# ── Online Training: Custom PPO Agents (run.py) ──────────────────────────────
# Supports: ppo, ppo_lstm, ppo_transformer

# Custom env (default)
uv run src/scripts/run.py --run_group=debug --base_config=ppo --seed=0

# FinRL env (richer observations: MACD, RSI, CCI, turbulence)
uv run src/scripts/run.py --run_group=debug --base_config=ppo --use_finrl --seed=0

# With differential Sharpe ratio reward (works with both custom and FinRL envs)
uv run src/scripts/run.py --run_group=debug --base_config=ppo --reward_type=diff_sharpe --seed=0
uv run src/scripts/run.py --run_group=debug --base_config=ppo --use_finrl --reward_type=diff_sharpe --seed=0

# With macro features (custom env only; requires FRED_API_KEY env var, falls back to zeros)
uv run src/scripts/run.py --run_group=debug --base_config=ppo --use_macro --seed=0

# ── Online Training: SB3 Baselines (run_sb3.py) ──────────────────────────────
# Supports: a2c, ppo_sb3, sac_sb3, td3, ddpg, tqc
# All algorithms verified on both custom and FinRL environments

uv run src/scripts/run_sb3.py --run_group=debug --base_config=td3 --use_finrl --seed=0
uv run src/scripts/run_sb3.py --run_group=debug --base_config=sac_sb3 --use_finrl --seed=0
uv run src/scripts/run_sb3.py --run_group=debug --base_config=a2c --use_finrl --seed=0
uv run src/scripts/run_sb3.py --run_group=debug --base_config=a2c --use_macro --seed=0  # custom env only

# ── Online Training: FinRL Native Baselines (run_finrl_drl.py) ────────────────
# Trains FinRL's DRLAgent for PPO, A2C, SAC, TD3, DDPG on FinRL env
# Uses chronological train/val/test split (2008–2020/2021/2022–2026)

uv run src/scripts/run_finrl_drl.py --run_group=debug --seed=0
uv run src/scripts/run_finrl_drl.py --run_group=debug --algos ppo sac --seed=0

# ── O2O Pipeline (run_o2o.py) ─────────────────────────────────────────────────
# Offline CQL → online SAC-Dirichlet

uv run src/scripts/run_o2o.py --run_group=debug --phase=o2o --seed=0
uv run src/scripts/run_o2o.py --run_group=debug --phase=o2o --use_mutual_funds --start_date=1995-01-01 --seed=0
uv run src/scripts/run_o2o.py --run_group=debug --phase=o2o --use_finrl_online --seed=0

# Offline CQL only
uv run src/scripts/run_o2o.py --run_group=debug --phase=offline --seed=0

# Online SAC-Dirichlet baseline
uv run src/scripts/run_o2o.py --run_group=debug --phase=sac --seed=0
```

### CLI Flag Reference

**`run.py`** (custom PPO agents):
`--base_config` (ppo|ppo_lstm|ppo_transformer), `--tickers`, `--start_date`, `--end_date`, `--train_ratio`, `--episode_length`, `--transaction_cost`, `--reward_type` (log_return|diff_sharpe), `--use_finrl`, `--finrl_time_window`, `--use_macro`, `--use_sentiment`, `--total_timesteps`, `--eval_interval`, `--n_eval_episodes`, `--lr`, `--n_steps`, `--batch_size`

**`run_sb3.py`** (SB3 baselines):
`--base_config` (a2c|ppo_sb3|sac_sb3|td3|ddpg|tqc), `--tickers`, `--start_date`, `--end_date`, `--train_ratio`, `--episode_length`, `--transaction_cost`, `--reward_type` (log_return|diff_sharpe), `--use_finrl`, `--finrl_time_window`, `--use_macro`, `--use_sentiment`, `--total_timesteps`, `--eval_interval`, `--n_eval_episodes`, `--save_model`

**`run_finrl_drl.py`** (FinRL native):
`--algos` (ppo|a2c|sac|td3|ddpg), `--tickers`, `--use_mutual_funds`, `--start_date`, `--train_end`, `--val_start`, `--val_end`, `--test_start`, `--end_date`, `--time_window`, `--transaction_cost`, `--total_timesteps`, `--eval_interval`, `--n_eval_episodes`, `--save_results`, `--save_models`

**`run_o2o.py`** (O2O pipeline):
`--phase` (offline|online|o2o|sac), `--use_finrl_online`, `--use_mutual_funds`, `--use_macro`, `--use_sentiment`, `--use_alpaca_embeddings`, `--n_offline_updates`, `--n_online_steps`, `--offline_data_steps`, `--tickers`, `--start_date`, `--train_end`, `--val_start`, `--val_end`, `--test_start`, `--end_date`, `--episode_length`, `--transaction_cost`, `--reward_type`

### Flag Compatibility Notes

- **`--reward_type=diff_sharpe`**: Works with both custom `PortfolioEnv` and FinRL env. When using FinRL, the wrapper overrides FinRL's default log-return reward with our differential Sharpe ratio computed from portfolio value changes. The `run_finrl_drl.py` script does NOT support `--reward_type` (FinRL-native training always uses log return).
- **`--use_macro`**: Appends 8 FRED macro features to the custom env observation vector. Supported by `run.py`, `run_sb3.py`, and `run_o2o.py` with the **custom env only**. Ignored with `--use_finrl` (a warning is printed). NOT supported by `run_finrl_drl.py`. FinRL has its own feature pipeline (MACD, RSI, CCI, turbulence) that is independent of our macro features.
- **`--use_sentiment`**: Appends SF Fed DNSI to the custom env observation vector. Same compatibility as `--use_macro`: custom env only, ignored with `--use_finrl`.
- **`--use_finrl`**: Supported by `run.py` and `run_sb3.py`. For `run_o2o.py`, use `--use_finrl_online` (offline phase always uses custom env).

### Online Learning Verification (1000-step smoke tests)

| Algorithm | Script | Custom Env | FinRL Env | diff_sharpe + FinRL |
|-----------|--------|:----------:|:---------:|:-------------------:|
| PPO | `run.py` | ✅ | ✅ | ✅ |
| PPO-LSTM | `run.py` | ✅ | ✅ | — |
| PPO-Transformer | `run.py` | ✅ | ✅ | — |
| A2C (SB3) | `run_sb3.py` | ✅ | ✅ | ✅ |
| PPO (SB3) | `run_sb3.py` | ✅ | ✅ | — |
| SAC (SB3) | `run_sb3.py` | ✅ | ✅ | — |
| TD3 (SB3) | `run_sb3.py` | ✅ | ✅ | — |
| DDPG (SB3) | `run_sb3.py` | ✅ | ✅ | — |
| TQC (SB3) | `run_sb3.py` | ✅ | ✅ | — |
| PPO (FinRL) | `run_finrl_drl.py` | — | ✅ | — |
| A2C (FinRL) | `run_finrl_drl.py` | — | ✅ | — |
| SAC (FinRL) | `run_finrl_drl.py` | — | ✅ | — |
| TD3 (FinRL) | `run_finrl_drl.py` | — | ✅ | — |
| DDPG (FinRL) | `run_finrl_drl.py` | — | ✅ | — |

## Architecture

### Training Scripts

| Script | Algorithms | Environment | Purpose |
|--------|-----------|-------------|---------|
| `run.py` | PPO, PPO-LSTM, PPO-Transformer | Custom or FinRL | Custom online agents |
| `run_sb3.py` | A2C, PPO, SAC, TD3, DDPG, TQC | Custom or FinRL | SB3 online baselines |
| `run_finrl_drl.py` | PPO, A2C, SAC, TD3, DDPG | FinRL only | FinRL-native baselines |
| `run_o2o.py` | Geodesic-CQL, SAC-Dirichlet, O2O | Custom (offline) + Custom/FinRL (online) | Offline, online, and O2O |

### Novel Algorithms (CS 285 Project Contributions)

**1. SAC-Dirichlet** (`src/agents/sac_dirichlet.py`, `src/networks/dirichlet_policy.py`)
Replaces the Gaussian-softmax policy with a true Dirichlet distribution on the portfolio simplex. `DirichletActor` outputs concentration parameters `α = softplus(f_θ(s)) + 1` (ensures α > 1 for unimodal distributions), enabling exact entropy computation on the simplex via PyTorch's `Dirichlet` distribution.

**2. Geodesic-CQL** (`src/agents/cql_geodesic.py`)
Conservative Q-Learning for offline pre-training that uses Fisher-Rao geodesic distance as the CQL penalty metric:
```
d_FR(w1, w2) = 2 * arccos(Σ √(w_i * w'_i))
L_CQL = L_Bellman + β * d_FR(μ_θ(s), a_behavior) * max(Q_policy - Q_behavior, 0)
```
The Fisher-Rao distance is the natural Riemannian metric on the probability simplex — more geometrically meaningful than L2 distance in logit space.

**3. Regime-Conditioned POMDP** (`src/networks/regime_encoder.py`)
GRU-based `RegimeEncoder` processes a rolling window of observations to produce regime belief state `h_t`. Both actor and critic are conditioned on `h_t`: `π(a|s,h)`, `Q(s,a,h)`. `RegimeConditionedActor/Critic` wrap `DirichletActor/DoubleCritic` and prepend the regime vector.

**4. O2O Adaptive Pipeline** (`src/agents/o2o_agent.py`)
Offline pre-training (Geodesic-CQL) → online fine-tuning (SAC-Dirichlet). Regime KL divergence between offline/online distributions adaptively scales the CQL weight: `sigmoid(λ * KL(h_offline || h_online))`. Low KL = safe to exploit; high KL = maintain conservatism.

### Environment Layer

**Custom PortfolioEnv** (`src/envs/portfolio_env.py`): Gymnasium env with random episode start times. Observation: `[current_weights, features.flatten()]` where features has 6 dims per asset (log_return, rolling_mean, rolling_std, RSI, MACD, Bollinger%B). Supports `--reward_type=diff_sharpe` for risk-aware training and `--use_macro`/`--use_sentiment` for multimodal observations. **Primary use**: offline dataset generation (random-start episodes required for offline RL). Also used for online training when `--use_finrl` is not set.

**FinRLPortfolioWrapper** (`src/envs/finrl_wrapper.py`): Subclass of `gymnasium.Env` adapting FinRL's old-gym `PortfolioOptimizationEnv` to the gymnasium interface. Flattens 3D observations `(n_features, n_stocks, time_window) → 1D`. Sanitizes inf/NaN from FinRL's "by_previous_time" normalization (replaces with 0). Translates FinRL's internal portfolio metrics (`_portfolio_value`, `_final_weights`) to our standard `info` dict keys (`portfolio_value`, `turnover`) so all evaluation code works uniformly. Supports `reward_type="diff_sharpe"` by overriding FinRL's log-return reward with differential Sharpe ratio computed from portfolio value changes. For DirichletActor (`accept_portfolio_weights=True`): converts `w → log(w)` before passing to FinRL so `softmax(log(w)) = w`. FinRL's constructor has a typo: `comission_fee_pct` (one 'm').

**ActionBoundedWrapper** (`src/envs/action_bounded_wrapper.py`): `gymnasium.Wrapper` that replaces unbounded `Box(-inf, inf)` action spaces with `Box(-10, 10)`. Required by ALL SB3 algorithms (SB3 2.x asserts finite bounds). Applied automatically by `run_sb3.py` and `run_finrl_drl.py`. Safe because `PortfolioEnv` clips logits to [-20, 20] before softmax. Works with both `PortfolioEnv` and `FinRLPortfolioWrapper` (both are `gymnasium.Env` subclasses).

**Key design decision**: Offline phase always uses the custom env (FinRL resets from data start; offline sampling requires random-start episodes). FinRL env is only used for the online phase.

### Data Flow & Evaluation Splits

**Ticker Universes:**
The project tests distribution shifts on an "All-Weather" macro portfolio. We support two 8-asset universes specifically chosen to represent orthogonal risk premia:
1.  **Standard ETFs (Orthogonal 8)**: `[SPY, EEM, TLT, HYG, DBC, GLD, UUP, SHY]` 
    *(Data begins 2008 due to UUP/DBC launch dates. Represents Equity, Emerging, Duration, Credit, Commodity, Gold, USD, and Cash proxies).*
2.  **Mutual Fund Proxies**: `[VFINX, VEIEX, VUSTX, VWEHX, PCRIX, USERX, VMFXX, VFISX]` 
    *(Triggered via `--use_mutual_funds`). These map 1:1 to the macro ETF exposures above (VMFXX serves as the USD/cash proxy for UUP) but allow `yfinance` to pull data back to the 1990s to capture the Dot-Com bubble while maintaining the exact 8-dimensional action space.*

**Chronological Splits (Critical for O2O Hypothesis):**
To systematically prove that pure Offline RL fails under structural distribution shifts and O2O successfully adapts, we strictly enforce the following chronological split:

* **Train (Offline Dataset)**: `2008-01-01` (or `1995-01-01` for mutual funds) to `2020-12-31`. 
    * *Rationale:* Exposes the offline agent to the 2008 GFC and the 2020 COVID-19 crash. In both events, stocks crashed while bonds/gold spiked. The offline agent will heavily overfit to this "flight to safety" correlation.
* **Validation (Hyperparameter Tuning)**: `2021-01-01` to `2021-12-31`. 
    * *Rationale:* A necessary 1-year buffer used *strictly* to tune learning rates, CQL alphas, and network sizes. In quantitative finance, chronological validation is mandatory to prevent look-ahead bias and protect the integrity of the test set.
* **Test (O2O Fine-Tuning Phase)**: `2022-01-01` to `2026-03-31`. 
    * *Rationale:* In 2022, rapid inflation and Fed rate hikes caused *both* stocks and bonds to crash simultaneously, breaking a 40-year market correlation. The frozen offline agent will confidently fail here (OOD overestimation), providing the exact distribution shift required to prove the value of the O2O algorithm's rapid online adaptation.

```text
yfinance → download_price_data(use_mutual_funds=True/False)
         → compute_features() [6 per-asset features: log_return, RSI, MACD, etc.]
         → [optional] download_fred_macro() [8 global features: DGS10, DFF, yield_spread,
                                              CPI_YoY, UNRATE, GDP_growth, UMich_sentiment, NFCI]
         → [optional] load_sentiment() [SF Fed DNSI scalar + Alpaca News embeddings (384-d)]
         → Chronological Split (Train: 2008-2020 | Val: 2021 | Test: 2022-2026)
         → PortfolioEnv (custom) or FinRLPortfolioWrapper
         → Agent training loop → WandB
```

Observation vector structure (flat 1D):
```
Custom env:  [weights(8), per_asset_TA(8×6=48), macro(8, if --use_macro), sentiment(1+, if --use_sentiment)]
FinRL env:   flattened (n_features × n_stocks × time_window) = e.g. (10 × 8 × 20) = 1600
```

### Config System

`src/configs/__init__.py` maps algorithm names to config modules (`CONFIG_MAP`). Each config returns an `ml_collections.ConfigDict`.

| Config | Module | Used by |
|--------|--------|---------|
| `ppo` | `ppo_config` | `run.py` |
| `ppo_lstm` | `ppo_lstm_config` | `run.py` |
| `ppo_transformer` | `ppo_transformer_config` | `run.py` |
| `sac_dirichlet` | `sac_dirichlet_config` | `run_o2o.py --phase=sac` |
| `cql_geodesic` | `cql_geodesic_config` | `run_o2o.py --phase=offline` |
| `o2o` | `o2o_config` | `run_o2o.py --phase=o2o` |
| `a2c` | `a2c_config` | `run_sb3.py` |
| `ppo_sb3` | `ppo_sb3_config` | `run_sb3.py` |
| `sac_sb3` | `sac_sb3_config` | `run_sb3.py` |
| `td3` | `td3_config` | `run_sb3.py` |
| `ddpg` | `ddpg_config` | `run_sb3.py` |
| `tqc` | `tqc_config` | `run_sb3.py` |

### Key Implementation Details

- **FinRL import**: Uses `importlib.import_module("finrl.meta.env_portfolio_optimization.env_portfolio_optimization")` directly to bypass FinRL's broken `__init__.py` (unconditionally imports `wrds`).
- **FinRL obs sanitization**: FinRL's `"by_previous_time"` normalization divides by previous timestep values, producing `inf` when values are near zero (e.g., at boundary time steps). `FinRLPortfolioWrapper._flatten_obs()` replaces non-finite values with 0.
- **FinRL info dict translation**: FinRL's `step()` returns `price_variation` and `trf_mu` instead of our standard `portfolio_value`/`turnover`. The wrapper reads `_portfolio_value` and `_final_weights` from the raw FinRL env to populate our standard info keys.
- **FinRL reward override**: FinRL internally computes log portfolio return. When `reward_type="diff_sharpe"` is set, `FinRLPortfolioWrapper` overrides FinRL's reward with differential Sharpe ratio computed from portfolio value changes. The `run_finrl_drl.py` script does not support this override (always uses FinRL's default).
- **torch constraint**: `torch>=2.0,<2.3` — Intel Mac (x86_64) has no wheel for torch 2.3+.
- **numpy constraint**: `numpy>=1.24,<2.0` — torch 2.2 requires numpy < 2.0.
- **Dirichlet sampling**: Uses reparameterized `dist.rsample()` via the Gamma trick for backpropagation through samples.
- **Target entropy**: `log(n_assets)` — entropy of uniform Dirichlet = max possible.
- **Replay buffer sequences**: `sample_with_context(batch_size)` returns `obs_seq (batch, seq_len, obs_dim)` for the regime encoder; `seq_len = regime_window` from config.
- **Macro features** (`src/envs/macro_features.py`): 8 FRED series (DGS10, DFF, yield_spread, CPI_YoY, UNRATE, GDP_growth, UMich_sentiment, NFCI). Requires `FRED_API_KEY` env var (free key at fred.stlouisfed.org). Falls back to zeros if key is absent so training still runs.
- **Sentiment features** (`src/envs/sentiment_features.py`): Two sources: (1) SF Fed Daily News Sentiment Index — auto-downloaded on first run, saved to `data/dnsi.xlsx`; (2) `AlpacaNewsEmbeddings` — fetches historical news via Alpaca API, encodes with `all-MiniLM-L6-v2` (384-d), caches to `data/alpaca_embeddings_*.pkl`. Requires `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`. Precompute once with `AlpacaNewsEmbeddings(tickers).precompute(start, end)`.
- **Multimodal obs dim**: base = `n_assets + n_assets×6`; add `+8` with `--use_macro`; add `+1` with `--use_sentiment`; add `+384` with `--use_alpaca_embeddings`.
- **SB3 progress bar**: Requires `rich` package (added as dependency).
