# CS285 Final Project — Consolidated Progress

**Project**: Offline-to-Online Deep RL for Portfolio Optimization
**Last updated**: 2026-04-09

This document consolidates `CLAUDE.md`, `IMPLEMENTATION_ROADMAP.md`, `plans/plan.md`, `ONLINE_RL_AUDIT.md`, and `TODO_GPU.md` into a single progress hand-off.

---

## 1. Project Hypothesis

The offline RL agent trains on **2008–2020** data, where stock crashes (2008 GFC, 2020 COVID) coincided with bond/gold rallies — the classic "flight to safety" correlation. In **2022**, rapid Fed rate hikes broke this 40-year correlation and *both* stocks and bonds crashed simultaneously.

**Claim**: Pure offline RL trained on 2008–2020 will fail on 2022+ due to OOD overestimation, while an O2O pipeline (offline pre-train → online fine-tune with adaptive conservatism) will adapt and outperform.

**Novel contributions**:
1. **SAC-Dirichlet** — true Dirichlet distribution on the portfolio simplex (not Gaussian-softmax), exact entropy via `torch.distributions.Dirichlet`.
2. **Geodesic-CQL** — CQL penalty using **Fisher-Rao** geodesic distance on the simplex: `d_FR(w₁,w₂) = 2·arccos(Σ √(w_i·w'_i))`.
3. **Regime-Conditioned POMDP** — GRU regime encoder produces belief state `h_t`; both actor and critic are conditioned on `h_t`.
4. **O2O Adaptive Pipeline** — KL divergence between offline/online regimes adaptively scales the CQL weight: `β_t = sigmoid(λ · KL(h_offline ‖ h_online))`.

---

## 2. Current Implementation Status

### 2.1 Algorithms Implemented (15 + 3 classical = 18)

| Category | Algorithm | Script | Status |
|----------|-----------|--------|--------|
| **Imitation** | BC | `run_offline.py` | ✅ Smoke-tested |
| **Imitation** | Fisher-BC | `run_offline.py` | ✅ Smoke-tested |
| **Actor-Reg** | TD3+BC | `run_offline.py` | ✅ Smoke-tested |
| **Actor-Reg** | AWAC | `run_offline.py` | ✅ Smoke-tested |
| **Conservative** | CQL (vanilla) | `run_offline.py` | ✅ Smoke-tested |
| **Conservative** | Geodesic-CQL ★ | `run_o2o.py --phase=offline` | ✅ |
| **Advantage** | IQL | `run_offline.py` | ✅ Smoke-tested |
| **Ensemble** | EDAC (N=10 critics) | `run_offline.py` | ✅ Smoke-tested |
| **Generative** | BCQ (VAE) | `run_offline.py` | ✅ Smoke-tested |
| **Model-based** | MBPO | `run_offline.py` | ✅ Smoke-tested |
| **Model-based** | MOPO | `run_offline.py` | ✅ Smoke-tested |
| **Sequence** | Decision Transformer | `run_offline.py` | ✅ Smoke-tested |
| **Sequence** | Trajectory Transformer | `run_offline.py` | ✅ Smoke-tested |
| **Online Novel** | SAC-Dirichlet ★ | `run_o2o.py --phase=sac` | ✅ |
| **O2O Pipeline** | O2O Agent ★ | `run_o2o.py --phase=o2o` | ✅ |
| **Online Custom** | PPO / PPO-LSTM / PPO-Transformer | `run.py` | ✅ |
| **Online SB3** | A2C, PPO, SAC, TD3, DDPG, TQC | `run_sb3.py` | ✅ |
| **Online FinRL** | PPO, A2C, SAC, TD3, DDPG | `run_finrl_drl.py` | ✅ |
| **Classical** | Equal Weight, Inverse Vol, Markowitz | `run_classical.py` | ✅ Smoke-tested |

### 2.2 Smoke Test Results (100-update runs, sanity-check only)

| Algorithm | Sharpe | Annual Ret | Max DD |
|-----------|-------:|-----------:|-------:|
| BC | 0.83 | 4.2% | 4.0% |
| Fisher-BC | 0.80 | 5.4% | 3.1% |
| IQL | 1.24 | 4.6% | 3.6% |
| AWAC | 1.29 | 7.5% | 3.1% |
| CQL Vanilla | 1.88 | 10.9% | 2.4% |
| TD3+BC | 1.97 | 10.8% | 2.9% |
| EDAC | 1.74 | 10.3% | 2.6% |
| BCQ | 1.59 | 7.1% | 3.1% |
| MBPO | 0.87 | 4.1% | 3.5% |
| MOPO | 1.49 | 7.8% | 3.1% |
| Decision Transformer | 2.70 | 14.5% | 2.4% |
| Trajectory Transformer | 1.54 | 8.1% | 3.3% |

> **These are NOT meaningful comparisons** — only 100 updates with 2k transitions. They merely confirm every script runs end-to-end without crashing.

### 2.3 Infrastructure

| Component | Location |
|-----------|----------|
| Standard replay buffer | `src/agents/replay_buffer.py:19` |
| N-step replay buffer | `src/agents/replay_buffer.py:225` |
| Context-window sampling (regime) | `src/agents/replay_buffer.py:68` |
| Trajectory segment sampling (DT/TT) | `src/agents/replay_buffer.py:168` |
| Return-to-go computation | `src/agents/replay_buffer.py:152` |
| Offline dataset generation | `src/agents/replay_buffer.py:112` |
| Chronological train/val/test split | `src/envs/data_utils.py` |
| Macro features (8 FRED series) | `src/envs/macro_features.py` |
| Sentiment features (DNSI + Alpaca) | `src/envs/sentiment_features.py` |
| Custom env (random-start episodes) | `src/envs/portfolio_env.py` |
| FinRL gymnasium adapter | `src/envs/finrl_wrapper.py` |
| SB3 action-space wrapper | `src/envs/action_bounded_wrapper.py` |

---

## 3. Architecture Overview

### 3.1 Training Scripts

| Script | Algorithms | Env |
|--------|-----------|-----|
| `run.py` | PPO / PPO-LSTM / PPO-Transformer | Custom or FinRL |
| `run_sb3.py` | A2C, PPO, SAC, TD3, DDPG, TQC | Custom or FinRL |
| `run_offline.py` | BC, Fisher-BC, TD3+BC, AWAC, CQL, IQL, EDAC, BCQ, MBPO, MOPO, DT, TT | Custom only |
| `run_classical.py` | Equal Weight, Inverse Vol, Markowitz | Custom |
| `run_finrl_drl.py` | PPO, A2C, SAC, TD3, DDPG (FinRL native) | FinRL only |
| `run_o2o.py` | Geodesic-CQL, SAC-Dirichlet, O2O | Custom (offline) + Custom/FinRL (online) |

### 3.2 Data & Splits

**Asset universes (8-asset all-weather portfolios):**
- **ETFs (default)**: `[SPY, EEM, TLT, HYG, DBC, GLD, UUP, SHY]` — data from 2008.
- **Mutual fund proxies** (`--use_mutual_funds`): `[VFINX, VEIEX, VUSTX, VWEHX, PCRIX, USERX, VMFXX, VFISX]` — data from 1995.

**Chronological splits**:
| Split | Period | Purpose |
|-------|--------|---------|
| Train (offline) | 2008-01-01 → 2020-12-31 | Offline dataset (GFC + COVID) |
| Validation | 2021-01-01 → 2021-12-31 | Hyperparameter tuning |
| Test | 2022-01-01 → 2026-03-31 | Distribution-shift evaluation |

**Observation vector** (custom env, flat 1D):
```
[weights(8), per_asset_TA(8×6=48), macro(8, optional), sentiment(1+, optional)]
```
- base = 56
- `+8` with `--use_macro`
- `+1` with `--use_sentiment`
- `+384` with `--use_alpaca_embeddings`

### 3.3 Key Run Commands

```bash
# Setup
uv sync --python 3.12
uv run wandb login

# Offline RL baselines
uv run src/scripts/run_offline.py --base_config=bc --run_group=baselines --seed=0
uv run src/scripts/run_offline.py --base_config=iql --run_group=baselines --seed=0
# ... see CLAUDE.md for full list

# Classical benchmarks
uv run src/scripts/run_classical.py --run_group=baselines --seed=0

# Novel: Geodesic-CQL → SAC-Dirichlet
uv run src/scripts/run_o2o.py --phase=offline --run_group=novel --seed=0
uv run src/scripts/run_o2o.py --phase=sac --run_group=novel --seed=0
uv run src/scripts/run_o2o.py --phase=o2o --run_group=novel --seed=0

# Online baselines
uv run src/scripts/run_sb3.py --base_config=sac_sb3 --use_finrl --seed=0
uv run src/scripts/run_finrl_drl.py --seed=0

# Disable wandb for local debugging
WANDB_MODE=disabled uv run src/scripts/run_offline.py --base_config=bc --seed=0
```

---

## 4. Mistakes Made / Lessons Learned

### 4.1 System-Level Audit Findings (`ONLINE_RL_AUDIT.md`)

**This is "simulated online RL", NOT true online RL.**
- All historical price data is loaded into memory at startup.
- The agent's learning loop is incremental, but the data source is fixed.
- Every observation the agent ever sees was available at `t=0`.
- The chronological train/val/test split is the only mechanism enforcing distribution shift.
- True online RL would require a streaming `MarketFeed`, an `OnlineTradingEnv` advancing with the market clock, and incremental feature computation. Not implemented (proposed in audit Section 5 as future work).

### 4.2 FinRL Integration Bugs Hit

| Bug | Fix |
|-----|-----|
| FinRL's `__init__.py` imports `wrds` unconditionally → import fails | Use `importlib.import_module("finrl.meta.env_portfolio_optimization.env_portfolio_optimization")` directly |
| FinRL's `"by_previous_time"` normalization divides by ~0 → produces `inf` | `FinRLPortfolioWrapper._flatten_obs()` replaces non-finite with 0 |
| FinRL's `step()` returns `price_variation`/`trf_mu`, not our `portfolio_value`/`turnover` | Wrapper reads `_portfolio_value`/`_final_weights` from raw env |
| FinRL constructor typo: `comission_fee_pct` (one 'm') | Match the typo when calling |
| FinRL `reset()` always starts from `_time_window` index — no random starts | Custom env is used for offline data collection; FinRL only for online phase |
| FinRL `PortfolioOptimizationEnv` uses old gym API | `FinRLPortfolioWrapper` adapts to gymnasium |

### 4.3 SB3 Compatibility Bugs

| Bug | Fix |
|-----|-----|
| SB3 2.x asserts finite action-space bounds; our env had `Box(-inf, inf)` | `ActionBoundedWrapper` clips to `Box(-10, 10)`; applied automatically by `run_sb3.py` and `run_finrl_drl.py` |
| SB3 progress bar requires `rich` | Added `rich` to `pyproject.toml` |

### 4.4 `--use_macro` Bugs

1. **`run_o2o.py --use_finrl_online --phase=o2o` crashes** — custom env (obs_dim=56 or 64 with macro) and FinRL env (obs_dim≈1600) have incompatible dimensions; the regime encoder GRU is sized for the offline obs but receives FinRL obs at online phase. **Fix**: script now blocks this combination and falls back to custom env with a warning. `--use_finrl_online` only works with `--phase=sac`.
2. **`run_finrl_drl.py --use_macro` silently degraded** — flag was parsed but only printed a warning. **Fix**: now raises `ValueError`.
3. `run.py`/`run_sb3.py --use_macro --use_finrl` prints a warning and proceeds without macro (acceptable; FinRL has its own feature pipeline).

### 4.5 Vanilla-CQL Implementation Mistake

When subclassing `GeodesicCQL` to make `VanillaCQLAgent`, initially called the regime-conditioned critic incorrectly:
```python
# WRONG: directly indexed internals + element-wise added regime
self.critic.critic.q1(torch.cat([obs_rep + regime_rep, w_policy], dim=-1))
```
The proper interface (which concatenates internally) is:
```python
q1_policy, q2_policy = self.critic(obs_rep, w_policy.detach(), regime_rep)
```

### 4.6 Environment Constraints

- `torch>=2.0,<2.3` — Intel Mac (x86_64) has no torch 2.3+ wheel.
- `numpy>=1.24,<2.0` — torch 2.2 requires numpy <2.0.
- `--use_macro` falls back to zeros without `FRED_API_KEY` — runs complete but the ablation is meaningless. **Always check the key is set before declaring macro results.**
- yfinance has rate limits — stagger seeds or pre-download.

---

## 5. To-Do

### 5.1 Critical Path (For Paper / Bayesian Extension)

The **only major missing algorithmic piece** is the Bayesian regime encoder proposed in the experimental plan. ~500 LOC across 5–6 files.

- [ ] **`BayesianRegimeEncoder`** (`src/networks/bayesian_regime_encoder.py`, NEW)
  - GRU + posterior `(μ, log σ)` head + learned prior `(μ_p, log σ_p)` head
  - `forward()` returns `regime_mean`, `regime_std`, `kl_loss`
  - `sample()` reparameterized; `encode_step()` for online rollout
- [ ] **`BayesianRegimeConditionedActor`** wrapping `DirichletActor` with Thompson sampling
- [ ] **`BayesianRegimeConditionedCritic`** taking regime samples as input
- [ ] **Uncertainty-weighted CQL penalty** (`src/agents/cql_geodesic.py`):
  ```
  β_t = β₀ · sigmoid(λ · KL(h_offline ‖ h_online) + γ · Var[h_t])
  ```
  Currently only `sigmoid(λ · KL)` — `γ · Var[h_t]` term is missing.
- [ ] **Risk-aware exploration** in `sac_dirichlet.py` `collect_step()` (Thompson sampling + temperature scaling by regime σ)
- [ ] **O2O integration** (`src/agents/o2o_agent.py`) — replace `RegimeEncoder` with Bayesian variant
- [ ] **`--bayesian` CLI flag** + `bayesian_o2o_config.py` + `bayesian_cql_geodesic_config.py`
- [ ] **Sharpe ratio** metric in evaluation (compute from per-step returns)
- [ ] **Maximum drawdown** metric in evaluation (running max portfolio value)

### 5.2 Important but Non-Blocking

- [ ] **Wire `--n_step` into `run_o2o.py`** — currently only `run_offline.py` supports it. `NStepReplayBuffer` exists but is not plugged into the O2O pipeline.
- [ ] **Adjust CQL Bellman target for n-step**: in `cql_geodesic.py`, change bootstrap discount from `γ` to `γ^n` (stored reward already accumulates `Σ γ^i r_i` from the n-step buffer).
- [ ] **Log regime KL divergence to WandB** during O2O fine-tuning (already computed in `o2o_agent.py:141-159`).
- [ ] **Uncertainty-weighted regret** metric: `regret_t = (oracle_return_t - agent_return_t) · σ_regime_t` (requires Bayesian encoder).
- [ ] **Regime posterior visualization** — plot `μ_t` and `σ_t` over the test period for figures.

### 5.3 Nice-to-Have / Research-Level

- [ ] Regime-conditioned dynamics: `P(s'|s,a,h_t)` — modify `EnsembleDynamics` to accept regime vector
- [ ] Bayesian GRU weights (Bayes by Backprop / Flipout)
- [ ] Posterior predictive rollouts (model-based + regime samples)
- [ ] True online RL: `MarketFeed` + `OnlineTradingEnv` + walk-forward `OnlineTrainer` (audit Section 5)

---

## 6. Experiment Execution Plan (GPU Hours)

### Phase 1 — Core Runs (P1, ~62 GPU-hours total)

3 seeds (0, 1, 2) per experiment.

| # | Group | Experiments | Approx GPU-h |
|---|-------|-------------|-------------:|
| 1a | `p1_novel` | O2O + Geodesic-CQL + SAC-Dirichlet | ~24 |
| 1b | `p1_finrl_compare` | PPO/A2C/SAC/TD3/DDPG (FinRL) | ~15 |
| 1c | `p1_sb3` | A2C, PPO, SAC, TD3, DDPG, TQC | ~18 |
| 1d | `p1_ppo` | Custom PPO MLP, 1M steps | ~5 |

### Phase 2 — Ablations (P2, ~40 GPU-h)

| Group | What |
|-------|------|
| `p2_mutual_funds` | O2O on 1995–2026 mutual fund proxies (Dot-Com coverage) |
| `p2_macro` | O2O + macro features (FRED) |
| `p2_finrl_online` | O2O with FinRL env for online phase |
| `p2_ppo_arch` | PPO-LSTM, PPO-Transformer |
| `p2_reward` | `diff_sharpe` vs `log_return` |
| `p2_cql_alpha` | Geodesic-CQL `α ∈ {1, 3, 5, 10}` (2 seeds is enough) |

### Phase 3 — Optional Extras (P3)

| Group | What |
|-------|------|
| `p3_sentiment` | O2O + SF Fed DNSI |
| `p3_multimodal` | O2O + macro + sentiment + Alpaca embeddings |
| `p3_finrl_mf` | FinRL compare on mutual fund universe |

### Phase 4 — Bayesian Experiments (after extension is implemented)

```bash
# Full Bayesian pipeline (main result)
uv run src/scripts/run_o2o.py --phase=o2o --bayesian --run_group=bayesian --seed=0

# Ablation: deterministic regime encoder (current system, already runnable)
uv run src/scripts/run_o2o.py --phase=o2o --run_group=bayesian_ablation --seed=0

# Ablation: Bayesian offline only / Bayesian online only
uv run src/scripts/run_o2o.py --phase=offline --bayesian --run_group=bayesian_ablation --seed=0
uv run src/scripts/run_o2o.py --phase=sac --bayesian --run_group=bayesian_ablation --seed=0
```

### One-Time Setup on a GPU node

```bash
uv sync --python 3.12
uv run wandb login
export FRED_API_KEY=...   # optional, for --use_macro
```

WandB project: `cs285-portfolio-rl`. SLURM templates for single jobs and array jobs are in `TODO_GPU.md` Section 5.

---

## 7. Milestone Report Status (4.2)

**Required**:
1. ½–1 page progress overview (one experiment described, hypothesis updates)
2. Training curve from a method that trains (need not be the final method)
3. Submit one PDF per group to Gradescope

**To-do for milestone**:
- [ ] `uv run wandb login` in this venv
- [ ] Run one full experiment (e.g., IQL or Geodesic-CQL, 50k updates) to produce a real WandB curve
- [ ] Screenshot the WandB loss / eval curves
- [ ] Write the ½–1 page summary covering: hypothesis, the chronological-split design, what's implemented (15 baselines + 4 novel components), the experiment performed, what's next (Bayesian extension)
- [ ] Export PDF, submit to Gradescope

---

## 8. Files Reference

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Authoritative reference for commands, CLI flags, configs, architecture |
| `plans/plan.md` | Experimental plan: hypothesis, splits, algorithms, ablations |
| `IMPLEMENTATION_ROADMAP.md` | Detailed implementation status + Bayesian extension tasks |
| `ONLINE_RL_AUDIT.md` | Technical audit of online-vs-simulated RL, FinRL limitations, `--use_macro` bugs |
| `TODO_GPU.md` | GPU experiment checklist with SLURM templates |
| `PROGRESS.md` | **This file** — consolidated hand-off |
