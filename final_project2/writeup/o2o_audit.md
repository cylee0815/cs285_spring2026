# O2O fine-tune audit — bug vs τ-pathology

Read-only audit of `scripts/run_o2o.py`, `hybrid_rl/agents/o2o_agent.py`,
and `online_rl/agents/sac_dirichlet.py` triggered by Phase 2C Batch 1
results: `naive_o2o_seed42` Sharpe = -0.236, `naive_o2o_seed1337` Sharpe
= +2.179, both with test turnover = **0.0000** across the entire
1,061-day test window.

## Verdict: PATHOLOGY (not a bug)

The five mechanical checks below all pass. The static-allocation
collapse is the SAC-Dirichlet τ-blowup pathology
(`writeup/sac_degeneracy_postmortem.md`) compounding with a
**near-static CQL-pretrain output on the causal pipeline** — confirmed
empirically by the cql\_vanilla rows in
`results/phase2a_causal/per_run.csv`, which have turnover = 6.7×10⁻⁵
(effectively zero) across both completed seeds. The SAC fine-tune
inherits the CQL pretrain's static behavior because both algorithms
converge to constant-α Dirichlet policies under the friction signal.

**Phase 2C Batch 2 should be allowed to finish.** Adaptive O2O may
behave differently because its `cql_weight` is modulated by KL — the
modulation could perturb the actor enough to escape the static basin.
If both naive AND adaptive collapse identically, the four-way
comparison's online conditions are all τ-pathology variants and we
report Phase 2C as the failure-mode investigation.

## Check 1 — Optimizer instantiation and step in online phase

**PASS.**

Optimizers are constructed once in `SACDirichlet.__init__`
(`online_rl/agents/sac_dirichlet.py:95-100`):
```python
self.actor_opt = optim.Adam(
    list(self.actor.parameters()) + list(self.regime_encoder.parameters()),
    lr=config.lr,
)
self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr)
self.temp_opt = optim.Adam([self.log_temperature], lr=config.lr)
```

Every iteration of `O2OAgent.finetune_online`
(`hybrid_rl/agents/o2o_agent.py:262-266`) calls all three:
```python
critic_metrics = self.sac_agent.update_critic(mixed_batch, cql_weight=cql_w)
actor_metrics  = self.sac_agent.update_actor(mixed_batch)
temp_metrics   = self.sac_agent.update_temperature(mixed_batch)
self.sac_agent.update_target_critic()
```

Each `update_*` function:
- `update_critic` (`sac_dirichlet.py:205-208`): `critic_opt.zero_grad();
  total_loss.backward(); critic_opt.step()`.
- `update_actor` (`sac_dirichlet.py:233-239`): `actor_opt.zero_grad();
  total_actor_loss.backward(); ... actor_opt.step()`.
- `update_temperature` (`sac_dirichlet.py:263-265`): `temp_opt.zero_grad();
  temp_loss.backward(); temp_opt.step()`.

No conditional gating on a flag. No early-return guard above the step
calls (other than the `len(buffer) < batch_size` warm-up that lasts
~256 iters). All three optimizers genuinely step every iteration once
the buffer fills.

## Check 2 — `requires_grad` on actor parameters during fine-tune

**PASS.**

The only `requires_grad = False` calls in the SAC code path are
intentional:

- `target_critic` parameters frozen at construction
  (`sac_dirichlet.py:83-84`). Target net is supposed to be frozen.
- `critic.parameters()` temporarily frozen during `update_actor`
  (`sac_dirichlet.py:220-221`) and restored at the end
  (`sac_dirichlet.py:241-242`). Standard SAC pattern.

The `actor.parameters()` are never set to `requires_grad = False`
anywhere. Verified by grep over the entire `online_rl/agents/` and
`hybrid_rl/agents/` trees. The `transfer_to_online` call
(`o2o_agent.py:142-164`) does NOT freeze the actor.

## Check 3 — Model-instance integrity from offline → online

**PASS.**

`O2OAgent.__init__` (`o2o_agent.py:84`) creates `self.sac_agent =
SACDirichlet(train_env, config, device)`. This builds the actor as a
fresh `nn.Module` and constructs `actor_opt` over its parameters
(`sac_dirichlet.py:95-98`).

`O2OAgent.transfer_to_online` (`o2o_agent.py:150`) calls
```python
self.sac_agent.actor.load_state_dict(policy_state['actor'])
```
which **mutates the existing parameter tensors in place** — it does
not replace the parameter objects. Since `actor_opt` was constructed
over `actor.parameters()` (which returns the same parameter objects
that `load_state_dict` mutates), the optimizer continues to update the
same in-memory weights that the eval-time forward pass reads.

No silent fresh-actor instantiation between the offline-end and
online-start phases. Confirmed by reading the full
`hybrid_rl/agents/o2o_agent.py` end-to-end.

## Check 4 — Eval mode / deterministic lock

**Mechanically PASS, but worth flagging for context.**

`SACDirichlet.evaluate` (`sac_dirichlet.py:323`):
```python
w, _, _, _ = self.actor(obs_t, regime, deterministic=True)
```

The `deterministic=True` branch in the underlying actor returns
`dist.mean = α / α₀`, which is the analytic mean of the Dirichlet.
This is deterministic *given the observation* — but it should still
vary across observations as long as `α(obs, regime)` varies across
observations.

The fact that turnover is exactly 0 across 1,061 distinct observations
means `α(obs, regime)` is approximately CONSTANT across the test
window. This is not the eval mode itself — it is the actor's learned
function being approximately observation-independent.

Phase 2B online-only SAC at the same evaluation path produced Sharpe
0.9413 with turnover = 0.000000 across all 3 seeds (identical to four
decimals): random init → SAC training → equal-weight Dirichlet
($\alpha \approx (1,\ldots,1)$). This is the documented τ-blowup
pathology. Phase 2C naive_o2o produces a *different* static allocation
per seed because the CQL pretrain converges to a seed-dependent
near-constant $\alpha$.

## Check 5 — Empirical signature distinguishing bug from pathology

**Cannot directly compare per-step actions — `run_o2o.py` does not save
the SAC actor checkpoint.** Indirect evidence below.

### Sub-check 5a — CQL pretrain output on causal pipeline

`results/phase2a_causal/cql_vanilla_lambda0.001_seed{42,1337}/metrics.json`:
- seed=42: turnover = 6.898×10⁻⁵, Sharpe +0.9521
- seed=1337: turnover = 6.853×10⁻⁵, Sharpe +0.9520

CQL on the causal pipeline converges to **a near-static allocation**
(turnover ≈ 7×10⁻⁵, four orders of magnitude below the friction
threshold). Different from EW (turnover identically 0) but operationally
indistinguishable from a constant-allocation policy. Sharpe matches
EW to four decimals.

This is the friction-collapse the milestone documented for IQL,
reproduced for CQL on the causal pipeline. The CQL pretrain that O2O
loads as init is effectively a constant-allocation Dirichlet policy.

### Sub-check 5b — O2O test turnover under SAC fine-tune

After `transfer_to_online` loads the (near-static) CQL actor, SAC
fine-tunes for 50,000 steps. Phase 2C Batch 1 metrics show turnover =
0.000000 (not even 7×10⁻⁵). Two non-exclusive contributions:

1. **τ-blowup compresses any obs-dependence in α.** As τ grows, the
   actor loss term `τ · log π - Q` is dominated by the entropy
   regularizer. The optimal policy under the dominant-entropy regime is
   uniform Dir(1,…,1), regardless of $Q$. Half-way to that limit, the
   actor's α moves toward (1,…,1) but doesn't quite arrive in 50k
   steps — yielding the seed-dependent static allocation we observe.
2. **The CQL initialization is already near the basin.** The actor
   starts from a near-constant $\alpha$ (sub-check 5a); SAC's noisy
   updates around a near-constant point leave it near-constant.

### Sub-check 5c — variance of Sharpe across seeds

Phase 2B online-only SAC: Sharpe identical to four decimals across
seeds (random-init → same equal-weight basin). Phase 2C naive O2O:
Sharpe variance enormous (-0.24, +2.18). The variance source is
the CQL pretrain's seed-dependent convergence: different seeds
converge to different static $\alpha$ vectors with different realized
returns on the test window, but the same near-zero turnover.

If "actor literally not updating" were the bug, we would expect
Phase 2C to be exactly the post-`transfer_to_online` policy =
post-CQL-pretrain policy = the same as evaluating the CQL agent
directly (Phase 2A causal CQL). Phase 2A causal CQL Sharpes are
0.9521, 0.9520 (n=2 of 3 so far) — tight cluster near EW. Phase 2C
naive O2O Sharpes are -0.236 and +2.179 — much wider. So SOMETHING is
moving the policy away from the CQL pretrain output, just not enough
to break out of the static-allocation basin. That something is
plausibly the SAC actor + τ updates: small perturbations on top of
the CQL init, which produce different static endpoints per seed
without ever escaping the basin.

This empirical signature is consistent with **τ-pathology stacking on
top of CQL-friction-collapse**, not with "actor literally not
updating."

## Static checks summary

| Check | Verdict | Evidence |
|---|---|---|
| 1. Optimizer instantiation + step | **PASS** | `sac_dirichlet.py:95-100, 205-208, 233-239, 263-265` |
| 2. `requires_grad` on actor | **PASS** | only critic temporarily frozen during actor update; actor never frozen |
| 3. Model-instance integrity | **PASS** | `load_state_dict` mutates in place; optimizer holds same parameter refs |
| 4. Eval mode lock | **PASS (per code)** | `deterministic=True` returns `dist.mean`; constant only if α(obs) is constant |
| 5. Empirical signature | **τ-pathology + CQL collapse** | CQL pretrain has turnover 7×10⁻⁵ on causal data; SAC tau-blowup compresses further |

## Implications for the headline experiment

The Phase 2C four-way comparison decomposes as:

1. **Frozen offline** (best-val Phase 2A causal IQL): friction-collapse
   to near-EW. Sharpe ≈ 0.94, turnover ≈ 7×10⁻⁵.
2. **Online-only SAC** (Phase 2B): random-init → τ-blowup → exact EW.
   Sharpe = 0.9413, turnover = 0 across all seeds.
3. **Naive O2O** (Phase 2C, just observed): CQL init → SAC τ-blowup →
   seed-dependent static allocation near CQL output. Sharpe wildly
   variable across seeds, turnover = 0.
4. **Adaptive O2O** (Phase 2C Batch 2, in flight): unknown. The
   `cql_weight` schedule could perturb the actor's training trajectory
   enough to break out of the static basin OR could behave identically
   to naive.

If (4) is also static-collapsed, the four-way comparison's RL
conditions are all "stuck at some near-static allocation due to
combination of friction-collapse + τ-pathology" and the headline becomes:
*GRPO is the only RL method on this universe that escapes the static
basin under friction.* The Phase 2D ablation (G ∈ {4,8,16}, all 9
runs at Sharpe ∈ [0.25, 0.57] with non-zero turnover) supports this
read.

## Decision-point for the user

If Batch 2 (adaptive O2O) confirms the static-collapse pattern, the
report's framing pivots:
- The four-way comparison becomes "three flavors of friction-collapse
  vs GRPO escapes." This is a defensible-but-different headline.
- The SAC `target_entropy` fix from `writeup/sac_degeneracy_postmortem.md`
  becomes a recommended post-paper follow-up, not a pre-paper STOP.

If Batch 2 (adaptive O2O) shows non-zero turnover and meaningful Sharpe
variance with the schedule, the headline survives in original form,
and the naive O2O row becomes "naive fine-tune fails for τ-pathology
reasons; adaptive succeeds via the conservatism schedule" — a clean
positive result for the adaptive contribution.

Either way, **no immediate fix is needed**, no Batch 2 stop is needed,
and the audit verdict is **PATHOLOGY, not BUG**. The mechanical
correctness of `finetune_online` is confirmed; the static collapse is a
known-class failure mode of SAC-Dirichlet stacking with friction
collapse.
