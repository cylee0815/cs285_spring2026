# Feature-leak triage — Phase 2A TD3+BC and BCQ anomaly is a label-leak

**Status: STOP-class finding. Pipeline contamination identified.
Phase 2C killed mid-flight before any contaminated metrics.json hit
disk. Phase 2A bcq / td3_bc rows reinterpreted as leak-exploitation
demonstrations, not algorithm baselines. No code patched; no re-runs
launched. Both decisions are the user's.**

## 1. Identified leak

The Phase 2A pipeline (`scripts/run_offline.py` →
`make_train_val_test_envs` → `core/envs/data_utils.py:compute_features`)
constructs the per-step `features` and `forward_returns` arrays such
that `features[t, asset, 0]` and `forward_returns[t, asset]` are the
*same financial signal*: today's return.

`core/envs/data_utils.py:compute_features` (relevant lines):
```python
# 136-138 — feature 0 is today's log-return
lr = np.log(close / close.shift(1))                # lr[t] = log(close[t] / close[t-1])
features[:, i, 0] = lr.values.astype(np.float32)
```
```python
# data_utils.py:268 — forward_returns built without shifting
forward_returns = np.expm1(log_returns).astype(np.float32)
```
After this, `features[t, asset, 0]` (the agent's observation at step t)
equals `log(1 + forward_returns[t, asset])` (the same-step reward
signal). The agent at decision time t has direct access to the return
its action will earn.

The PortfolioEnv contract assumes the opposite: per
`core/envs/portfolio_env.py:30-31`,

> ``forward_returns[t]`` is the asset return vector *after* the agent
> acts on ``features[t]``.

For that contract to hold, `forward_returns[t]` should equal
`prices[t+1]/prices[t] - 1`, i.e. the *next* day's return, not today's.
The Phase 2A pipeline violates this by one timestep.

The online-data pipeline (`features/feature_engineering.py:build_features`,
used by `scripts/build_real_dataset.py` and consumed by
`scripts/run_online_baselines.py`, `scripts/train_grpo.py`, etc.)
implements the contract correctly:
```python
# feature_engineering.py:178 — log_returns same as offline
log_returns[1:] = np.log(prices_arr[1:] / prices_arr[:-1])
```
```python
# feature_engineering.py:219-220 — but forward_returns ARE shifted
forward_returns = np.full_like(prices_arr, np.nan)
forward_returns[:-1] = prices_arr[1:] / prices_arr[:-1] - 1.0
```
Note the `[:-1] = prices_arr[1:] / prices_arr[:-1]` indexing: the agent
at decision time t earns the return between `prices[t]` and `prices[t+1]`.
Causal. The features at the same `t` use lagged log_returns up to
`log_returns[t]` (the realized return through close of day t), which
the agent observes at the close — standard end-of-day-decision setup.

**The bug is one missing alignment in `data_utils.py:compute_features`'s
caller**: `forward_returns` should be built with the same one-step
shift that `feature_engineering.py:219-220` performs. The header
docstring of PortfolioEnv (line 30-31) describes the correct contract
that the online pipeline satisfies and the offline pipeline violates.

## 2. Differential exploitation by algorithm class

Phase 2A grid (24 runs, 6 algorithms × seeds, λ=0.001 unless flagged).
Numbers are mean ± std across 3 seeds, from
`results/phase2a/per_run.csv`:

| algorithm | class | mean Sharpe | std | mean turnover | EW reference |
|---|---|---|---|---|---|
| **TD3+BC** | Q-max + soft BC penalty | **+6.785** | 0.628 | 0.644 | 0.95 |
| **BCQ** | Q-max + VAE candidate filter | **+3.905** | 0.232 | 0.263 | 0.95 |
| BC | pure imitation (no critic) | +0.941 | 0.006 | 0.003 | 0.95 |
| AWAC | adv-weighted regression to behavior | +0.942 | 0.011 | 0.004 | 0.95 |
| CQL (vanilla) | Q-max with conservative penalty | +0.945 | 0.001 | 0.0003 | 0.95 |
| IQL (λ=0)   | expectile + adv-weighted regression | +0.973 | 0.026 | 0.007 | 0.95 |
| IQL (λ=0.001) | expectile + adv-weighted regression | +0.943 | 0.002 | 0.003 | 0.95 |
| IQL (λ=0.005) | expectile + adv-weighted regression | +0.922 | 0.003 | 0.002 | 0.95 |

Pattern: the **two algorithms with the strongest critic-driven
allocation signal and the weakest behavior-policy anchor** (TD3+BC,
BCQ) escape the equal-weight basin and produce 3–7× over-EW Sharpe,
with non-trivial turnover (0.26 / 0.64). The other four
(BC/AWAC/CQL/IQL) are all behavior-anchored: BC is pure imitation,
AWAC and IQL fit advantage-weighted-regression toward behavior actions,
CQL adds a Q-pessimism penalty against OOD actions. None of these can
exploit the leak because exploiting it requires moving *away* from the
behavior policy (toward concentrated single-asset bets that the
behavior mixture — Dirichlet, EW, momentum, RP — never makes), and
their training objective forbids such drift.

The leak's *signature* is the algorithm-class differential, not just
the absolute level. CQL_vanilla in particular is interesting — same
critic-driven structure as TD3+BC, but its conservative-Q penalty
keeps it pinned at EW. That single-row discrimination is what makes
the leak hypothesis testable against the alternative "TD3+BC simply
found a real edge."

(Sanity check on leak realism: a deterministic strategy that places
100% on the next-day winning asset earns roughly 1% / day on this
universe, which annualizes to a Sharpe of ~7 at a daily-vol of ~1%.
The TD3+BC +6.785 number lands in that exact regime.)

## 3. Pipeline contamination map

| Phase | Pipeline | Caller | Status |
|---|---|---|---|
| 2A | `compute_features` (data_utils.py) | `run_offline.py:make_train_val_test_envs` | **leaky** |
| 2B | `build_features` (feature_engineering.py) | `run_online_baselines.py` (loads `datasets/real_*.npz`) | clean |
| 2C | `compute_features` (data_utils.py) | `run_o2o.py:188 make_train_val_test_envs` | **leaky** (killed before metrics.json) |
| 2D | `build_features` (feature_engineering.py) | `train_grpo.py` (loads `datasets/real_dirichlet.npz`) | clean |

Behavior-policy buffer construction (Phase 2A's offline mixture) is
*also* fed by the leaky env, but the behavior policies (Dirichlet
random, EW constant, momentum on past-window, RP on past-vol) do not
themselves consume the leaky feature in a way that exploits it; they
generate diverse non-cheating actions. The buffer's `(s, a, r)` rows
are correctly self-consistent under the leaky env's reward formula.
The leak only manifests when an algorithm conditions its action on the
leaky feature *and* is permitted to drift from the behavior support
(Q-maximizing without conservative anchoring).

Classical baselines (`evaluation/baselines.py` if present, or the EW
reference in the aggregator) read forward returns directly without
going through an RL agent — their Sharpe numbers are not affected by
the leak so long as they don't condition on `features[t, :, 0]`.

## 4. One-line fix (do not apply)

The fix mirrors what `feature_engineering.py:219-220` already does:
shift the forward-returns array by one timestep (so step-t reward uses
the next-day return) and trim the final row.

In `core/envs/data_utils.py`, around line 267-273 (or the analogous
block in `make_train_val_test_envs` around line 371-391):

```python
# BEFORE (leaky):
flat_features = _build_flat_features(features, macro_arr, sentiment_arr)
forward_returns = np.expm1(log_returns).astype(np.float32)

# AFTER (causal — matches feature_engineering.py:219-220 contract):
flat_features = _build_flat_features(features, macro_arr, sentiment_arr)
# forward_returns[t] is the return REALIZED on day t+1 — the return
# the agent's action at time t will earn. Strip the final row from
# both arrays (forward_returns[T-1] is undefined since there is no
# prices[T]).
fwd_full = np.expm1(log_returns).astype(np.float32)            # (T, N)
forward_returns = np.empty_like(fwd_full[:-1])                 # (T-1, N)
forward_returns[:] = fwd_full[1:]                              # shift by +1
flat_features = flat_features[:-1]                              # trim to match
log_returns_trim = log_returns[:-1]                             # for any downstream consumer
```

The same change must be made in *both* the
`make_train_test_envs` (line 267-268) and `make_train_val_test_envs`
(line 371-372) callsites — they share `compute_features` but each
constructs `forward_returns` independently and slices their own splits.

A defensive complement: add an assert in `PortfolioEnv.__init__` that
checks for the leak signature (per-asset correlation of
`features[:, i*F]` against `np.log1p(forward_returns[:, i])`); fail
loudly above some threshold (0.99) so this regression is caught at
env-construction time. Roughly 8 lines.

**Do not apply this patch unilaterally.** The aggregator currently
reads `results/phase2a/*/metrics.json` from runs trained against the
leaky env; applying the fix and re-running invalidates the existing
24-run table. The decision on whether to (a) re-run Phase 2A from
scratch under the fix, (b) port `run_o2o.py` to load
`datasets/real_dirichlet.npz` like the online runners do, or (c) leave
the offline pipeline broken and present the leak itself as a finding
is the user's to make after reading this report.

## 5. Methodological observation — class-differential as a leak diagnostic

The cleanest diagnostic for label/feature leakage in an offline-RL
benchmark is the *spread between behavior-anchored and Q-maximizing
algorithms on the same dataset*. With a well-constructed dataset:

- Imitation-anchored methods (BC, AWAC, IQL with high β-anchor, CQL with
  large pessimism coef) should approximate the behavior policy's mean
  performance. They cannot exceed the behavior policy by much.
- Q-maximizing methods with weak anchoring (TD3+BC, BCQ) should
  approximate or modestly exceed the behavior policy if the critic
  generalizes; they cannot dramatically exceed it absent OOD-Q
  exploitation, which on a clean dataset is exactly what the
  conservative-Q families exist to suppress.

A *large* gap (TD3+BC dramatically beating BC/AWAC/IQL on the same
buffer at the same evaluation split) is the load-bearing observation
for "the dataset has a label leak." Specifically:

- If TD3+BC ≈ BC ≈ EW: friction collapse / no exploitable signal.
- If TD3+BC > BC by a factor > 2 with non-trivial turnover, **and
  CQL_vanilla ≈ BC** (suggesting the conservative penalty is the
  diagnostic blocker, not the absence of signal): label leak.

The Phase 2A grid happens to instantiate exactly this differential.
TD3+BC outperforms BC by 7×; CQL_vanilla matches BC. Concentrated
mass (turnover 0.64) in TD3+BC vs. literal-EW (turnover 0.0003) in CQL
points at the OOD action class the leak rewards. The Phase 2A grid
is, post-hoc, a near-perfect leak detector — and its diagnostic value
is largely independent of the headline GRPO/O2O results.

This generalizes: any offline-RL benchmark builder should run TD3+BC
(or any pure Q-maximizing baseline with low BC weight) alongside a
strong behavior-anchored baseline (BC or AWAC) on every dataset, and
treat a >2× Sharpe gap as a STOP for label-leak inspection. Treat the
algorithm-class differential as a contract-violation signal of the
dataset, not as a method comparison.

This framing is a candidate Discussion-section contribution rather than
a Method-section claim: "we exploited (post-hoc) the BC ↔ TD3+BC
differential to detect a feature-construction bug in our offline
pipeline and recommend the differential as a routine pre-flight in
offline-RL benchmark construction."

## Operational state at time of writing

- **Phase 2A**: 24 runs on disk in `results/phase2a/`. BC/AWAC/CQL/IQL
  rows are valid as "behavior-anchored algorithms on a leaky env;
  behavior-anchor blocked the leak; all collapse to EW under λ=0.001."
  TD3+BC and BCQ rows are the leak demonstration; flag in writeup
  rather than re-cite as algorithm performance.
- **Phase 2B**: 9 runs on disk, clean pipeline. GRPO test Sharpe ~0.30
  across 3 seeds; PPO-LSTM has degenerate turnover (separate concern,
  not a leak); SAC has the documented τ-explosion pathology.
- **Phase 2C**: 0 runs on disk. Killed mid-batch-2; the in-flight
  Python processes died before writing any metrics.json. The
  aggregator's `[phase2c] no runs found at results/phase2c` confirms
  zero contamination.
- **Phase 2D**: 3 GRPO ablation runs on disk (G=4, 8, 16, all seed 42),
  causal pipeline. Sharpe 0.55 / 0.31 / 0.25 — *decreasing* in G. This
  is its own ablation finding (counter to pre-registered prediction)
  and unrelated to the leak.
- **Aggregator**: ran successfully, produced `results/phase2_summary.csv`
  with 33 rows (24 + 9). `results/phase2c/per_run.csv` does not exist
  (correctly skipped).
- **Plotter**: ran (`figures written to figures/phase2`); the
  four-way-comparison plot will only have the rows it has — frozen
  offline (Phase 2A best-val), online-only SAC (Phase 2B SAC), no
  naive O2O, no adaptive O2O. The plot will need re-running after
  the user-decided post-fix re-run.

## Decision tree for the user (do not pre-empt)

1. **Patch?** Apply the one-line fix in
   `data_utils.py:compute_features` (Section 4). Mechanical change.
2. **Re-run Phase 2C only against the fix?** Cheapest option: 6 runs,
   ~80 min wall-clock. Phase 2A's leak-exploitation findings remain
   valuable as a diagnostic.
3. **Re-run Phase 2A as well?** Adds ~25 min wall-clock for the 12
   non-Q-max rows (BC/AWAC/CQL/IQL); 6 leak-exploiting rows (BCQ,
   TD3+BC) become "valid baselines on causal pipeline" instead of
   "leak demonstrations" — different report framing.
4. **Framing**: lead Discussion with the leak-detection methodology
   (Section 5), or treat it as Limitations / Appendix?

The Phase 2 report's centerpiece (the four-way O2O comparison) is on
hold until decision 1 + 2 land.
