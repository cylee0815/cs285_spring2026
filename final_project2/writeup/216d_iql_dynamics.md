# 216-d IQL learning dynamics — verdict: under-convergent / monotonically degrading

## What I instrumented

Re-ran `scripts/run_iql_216d.py` with no hyperparameter changes (seed=42,
20k updates, default `iql_tau=0.7`, `iql_beta=3.0`, `lr=3e-4`,
`batch_size=256`) and added a 19-snapshot diagnostic schedule at every
1000 updates capturing:

- per-step training losses (value, critic, actor)
- IQL-internal signals (advantage_mean, weight_mean (post-clamp),
  policy entropy)
- val backtest (single chronological sweep over 2021, 252 days)

Output: `results/aux_iql_216d_diag/iql_216d_diag_seed42/diag_history.json`.
Run is deterministic; the `_backtest` calls during training use
`no_grad` and do not perturb the trained actor (verified: terminal test
Sharpe +0.3979 matches the original non-instrumented run exactly).

## Trajectory

```
step   val_sharpe  val_turn   v_loss   q_loss   actor_loss  adv_mean  weight_mean
 1000     +1.321    0.004    0.0000   0.0000    -8.5336    -0.0001     1.0000
 2000     +1.223    0.018    0.0000   0.0001    -8.5647    -0.0002     0.9999
 3000     +0.966    0.071    0.0000   0.0001    -8.6333    -0.0002     0.9999
 4000     +0.497    0.134    0.0000   0.0001    -8.7175    -0.0002     0.9999
 5000     +0.052    0.185    0.0000   0.0001    -8.8819    -0.0002     0.9999
 7000     -0.287    0.229    0.0000   0.0001    -8.8730    -0.0009     0.9997
10000     -0.592    0.276    0.0000   0.0001    -9.0352    -0.0007     0.9998
15000     -0.785    0.320    0.0000   0.0001    -9.1819    -0.0007     0.9998
19000     -0.946    0.333    0.0000   0.0001    -9.0411    -0.0009     0.9997
```

## Interpretation

### 1. The basin IS reachable, just not a fixed point

At step 1000 the val Sharpe is **+1.32** with turnover **0.004** — that
is firmly inside the behavior-anchored equal-weight basin (val EW
reference for 2021 is approximately 1.0; turnover < 0.005 means
near-static). The 216-d feature space supports the leak-invariance
basin at sufficient regularization, contrary to my earlier framing
("breaks in 216-d").

By step 5000 the val Sharpe has decayed to +0.05 with turnover 0.18.
By step 19000 it is −0.95 with turnover 0.33. The trajectory is
**monotonic in both direction and rate**: there is no oscillation, no
recovery, no plateau. Continuing past 20k would almost certainly
produce more of the same.

### 2. The drift is NOT critic-driven

Throughout training:

- `iql/value_loss` ≈ 0.0000 (V is essentially perfect on the offline
  buffer because Q is also essentially perfect and the expectile loss
  collapses)
- `iql/critic_loss` ≈ 0.0001 (Q has converged to predicting reward,
  which is small on this universe)
- `iql/advantage_mean` ≈ −0.0005 (the per-batch mean advantage is
  essentially zero, sometimes mildly negative)
- `iql/weight_mean` ≈ 0.9999 (post-clamp; the unclamped value is also
  near 1 because `exp(advantage / beta) = exp(0/3) = 1`)

The advantage-weighted-BC reduces to **plain BC with weights ≈ 1.0
across the buffer** for the entire training run. The drift away from
the behavior basin is therefore not a critic-driven "escape toward
high-Q regions" — it is the **policy slowly fitting the 216-d feature
space's idiosyncratic structure** under maximum-likelihood imitation
of the behavior actions.

What distinguishes this from BC on 56-d (which lands at val Sharpe ≈
0.94 and stays) is the feature dimensionality. With 6 lagged returns +
multi-window momentum + rolling vol per asset = 216-d, the policy has
enough capacity to express *non-trivial obs-dependent allocation rules
that match the behavior policy's per-state action variance*. The
behavior policy is a 4-way Dirichlet/EW/Momentum/RP mixture sampled
per-episode; on 216-d, the policy can learn "in this market regime,
allocate like Momentum; in that regime, allocate like RP; ..." The
same mixture on 56-d cannot be disambiguated to that resolution and
the policy collapses to the marginal mean (≈ EW).

### 3. Test ≠ Val on this drift

Terminal val Sharpe = −0.95 (deep negative). Terminal test Sharpe =
+0.40 (modestly positive). These are 1061-day vs 252-day windows in
adjacent calendar regimes (val = 2021, test = 2022 onward).

The 2021 val window is the COVID-recovery low-vol bull market; the
2022+ test window is the rate-shock + 2022 stock-bond correlation
break. An active-trading policy trained on 2008-2020 (which covers
GFC, taper tantrum, COVID crash) plausibly performs better in the
high-vol post-2021 regime than in the low-vol recovery regime. The
val/test divergence is consistent with regime-shift effects rather
than with any IQL-internal pathology.

### 4. Implication: the basin is a TRANSIENT state in 216-d, a FIXED POINT in 56-d

This is the substantive finding. On 56-d, IQL converges TO the EW
basin and stays. On 216-d, IQL TRANSITIONS THROUGH the EW basin
en route to a feature-driven active-trading regime. Both behaviors
are "leak-invariant" in the sense that no implausible Sharpe is
achieved (we never see TD3+BC's +6.8); the difference is in the
*dynamics around* the basin, not in the basin's existence.

A behavior-anchored IQL on 216-d with explicit early-stopping at the
val-Sharpe peak (around step 1000-2000) WOULD reproduce the basin
behavior. The default "20k updates" budget inherited from the 56-d
Phase 2A spec is, on 216-d, well past the val-Sharpe peak and into
the degrading regime.

## Verdict for the operational decision

**Under-convergent.** The trained-to-20k 216-d IQL is not at a stable
fixed point; it is at a particular point along a monotonically
degrading trajectory. Warm-starting GRPO from this point would
inherit the policy's "transitioning out of the EW basin" mid-trajectory
character, neither the early-training basin nor a converged
post-basin policy. The warm-start hypothesis (does GRPO benefit from
an offline-RL prior?) cannot be cleanly tested from this source.

The user's option 1 (skip GRPO warm-start, document as future work)
remains the right call. The dynamics check changes the *appendix
framing* of the 216-d IQL: rather than "leak-invariance breaks on
216-d," the cleaner statement is **"leak-invariance is feature-space
specific in its dynamics: on 56-d, the EW basin is a fixed point; on
216-d, it is a transient state on a longer trajectory toward
feature-driven active trading. Both are consistent with the
no-implausible-Sharpe leak-invariance claim, but only the 56-d case
holds at the standard 20k-update training budget."**

## Side notes

- An early-stopped 216-d IQL (best-val checkpoint) is plausibly a
  *better* GRPO warm-start source than the terminal one. We did not
  pursue this — it would multiply experimental conditions and the
  20k-update terminal point is what Phase 2A uses by spec — but it is
  a clean follow-up.
- The dynamics check is read-only in the sense that it reads the
  same trained model the original sanity-fail run produced (terminal
  test Sharpe is bit-identical, +0.3979). The instrumentation only
  added side-channel observation; it did not change the trained
  policy.
- The actor.pt at `results/aux_iql_216d_diag/iql_216d_diag_seed42/`
  is interchangeable with the actor.pt at
  `results/aux_iql_216d/iql_216d_seed42/` (same seed, same updates,
  same hyperparameters → bit-identical weights). Either may be kept;
  prefer the diag one for the appendix because its `diag_history.json`
  documents the trajectory.
