# Implementation notes — pre-existing bugs found and fixed during Phase 2 prep

Four bugs surfaced during smoke testing of the existing Phase-2 code paths
before any new training runs were launched. Each is fixed in the
`phase2-writeup` branch and was in place before any of the Phase 2A / 2B /
2C / 2D runs reported in this document started. Lines are quoted at
post-fix locations; pre-fix locations differ by ≤5 lines for the
non-deleted bugs.

---

## 1. `MomentumPolicy.get_action_from_returns` collapsed to equal weight

**File:** `policies/behavior.py:54-65`

**Pre-fix code:**
```python
window = returns_history[-self.lookback:]
scores = np.mean(window, axis=0)
scores_shifted = scores - np.max(scores)
exp_scores = np.exp(scores_shifted)
weights = exp_scores / exp_scores.sum()
```

**Failure mode.** Daily forward-return magnitudes for the 8-ETF basket are
~1e-4. Taking the mean over a 60-day lookback gives scores in the same
range. After `scores - max(scores)` they sit in `[-1e-3, 0]`, so
`exp(scores)` is in `[0.999, 1.000]` for every asset. The softmax is
numerically uniform: `weights ≈ [1/N, 1/N, ..., 1/N]`. The momentum policy
emitted equal-weight allocations for the entire dataset.

**Detected by.** Sanity-checking the four `datasets/real_*.npz` builds.
`real_momentum.npz` had `actions.mean(0) = [0.125]*8` and reward stats
(mean=0.00021, std=0.00516) bit-identical to `real_equal_weight.npz`. With
4587 transitions over 18 years that is statistically impossible for a
genuinely momentum-tilted policy.

**Fix.** `scores = np.sum(window, axis=0)` (60-day cumulative log return,
typical magnitude ~1e-2 to 1e-1, gives meaningful softmax spread). One
line. Verified post-fix: per-row max-min spread of the saved actions
went from 0.0 to 0.023 with non-zero per-coordinate std (~1%).

**Integrity impact.** The milestone narrative claims the offline behavior
distribution was a **mixture of Dirichlet / equal-weight / momentum /
risk-parity**. Pre-fix, the momentum component was indistinguishable from
the equal-weight component, so the mixture was effectively three policies
with one duplicated. Behavior coverage in the offline dataset was
narrower than reported. The Phase 2 `default_offline_mixture` (in
`policies/mixture.py`) loads the post-fix policy, so the new offline runs
are trained on the corrected mixture. The pre-Phase-2 27-IQL ablation
predates this finding entirely — its uniform-Dirichlet behavior buffer
(see Bug 4 of the persona note) sidesteps the problem by not invoking
`MomentumPolicy` at all.

---

## 2. `run_o2o.py` shadowed `O2OAgent` import inside `main()`

**File (pre-fix):** `scripts/run_o2o.py:309` had a redundant
`from hybrid_rl.agents.o2o_agent import O2OAgent` inside the
`elif args.phase == "offline":` branch, in addition to the canonical
top-of-file import at `scripts/run_o2o.py:45`.

**Failure mode.** Python's name-resolution rule: if any statement inside a
function rebinds a name, that name is treated as local for the entire
function body, regardless of textual order. So when `main()` reached
`agent = O2OAgent(custom_train_env, test_env, config, device)` on line
~270 in the `o2o` branch, the local-binding promise had already been
made by the import inside the unreached `offline` branch — and the
local had not yet been assigned. Result: `UnboundLocalError: cannot
access local variable 'O2OAgent' where it is not associated with a
value` at the first instruction of the o2o phase.

**Detected by.** First O2O smoke test (`run_o2o.py --phase=o2o`).

**Fix.** Delete the redundant inline import (`scripts/run_o2o.py` line 309
at HEAD now contains a docstring comment instead). One line.

**Integrity impact.** Pre-fix, **the `o2o` phase could not run at all**.
Any prior milestone result attributed to the O2O pipeline either came
from the `phase=offline` branch (which doesn't trigger the shadow) or
was never produced. This bug is interlocked with Bug 3 below: even after
fixing the import, the `o2o` branch crashed on its first online step.
Together, Bugs 2 and 3 indicate the o2o phase had not been end-to-end
exercised in CI before the Phase 2 prep work.

---

## 3. `O2OAgent.transfer_to_online` left the SAC env in a terminal state

**File:** `hybrid_rl/agents/o2o_agent.py:142-164`

**Pre-fix behavior.** `O2OAgent.__init__` calls
`self.sac_agent = SACDirichlet(train_env, config, device)`
(line 84), and `SACDirichlet.__init__` immediately runs
`self._obs, _ = env.reset()` (`online_rl/agents/sac_dirichlet.py:109`) to
cache the first observation. Subsequently, `agent.load_offline_data(...)`
calls `self.offline_buffer.load_from_env(self.train_env, ...)` — which
shares the same env object — and rolls it forward to populate the offline
buffer, leaving `train_env._t` at an exhausted/terminal index. The cached
`sac_agent._obs` is now stale.

When `transfer_to_online` finishes copying parameters and the next
`sac_agent.collect_step()` runs, line 147 of `sac_dirichlet.py` calls
`self.env.step(action)` on the still-exhausted env, which raises
`RuntimeError: Episode is done. Call reset() before stepping.`
(`core/envs/portfolio_env.py:142`).

**Detected by.** First successful run past Bug 2: the O2O smoke crashed
immediately on entering Phase 2.

**Fix.** Append four lines to `transfer_to_online`:
```python
self.sac_agent._obs, _ = self.sac_agent.env.reset()
self.sac_agent._episode_start = True
self.sac_agent._episode_return = 0.0
self.sac_agent._gru_hidden = self.sac_agent.regime_encoder.init_hidden(
    1, self.sac_agent.device
)
```
This re-anchors the SAC agent on a fresh post-reset env state, matching
the contract the rest of `collect_step` expects.

**Integrity impact.** Same as Bug 2: pre-fix, the o2o phase could not
make a single online gradient step. Any reported O2O result must have
post-dated this fix.

---

## 4. `run_o2o.py` o2o-phase ignored `cql_weight` — the centerpiece bug

**File (pre-fix):** `scripts/run_o2o.py:307-326`, an inline online-phase
loop that called
```python
m.update(agent.sac_agent.update_critic(mixed))
```
without forwarding any `cql_weight` argument.

**File (post-fix):** `scripts/run_o2o.py:307-312`, replaced by a call to
```python
o2o_history = agent.finetune_online(n_online)
```

**Failure mode.** `SACDirichlet.update_critic`'s signature is
`def update_critic(self, batch, cql_weight: float = 0.0)`
(`online_rl/agents/sac_dirichlet.py:173`). The pre-fix inline loop never
passed the kwarg, so the critic was updated with `cql_weight=0.0` for
the entire online phase — i.e., **no conservatism** in any iteration,
regardless of the `--adaptive_conservatism` CLI flag that we added in
this branch (and regardless of the original config's `cql_alpha=5.0`).

The full `O2OAgent.finetune_online` method
(`hybrid_rl/agents/o2o_agent.py:204-275`) implements the adaptive
schedule:
- Tracks recent online regime samples (lines 225-241).
- Computes `cql_w = self._compute_adaptive_cql_weight(h_online, var_online)`
  via `α_CQL · σ(η · KL)` (line 247, formula at lines 191-200).
- Calls `update_critic(mixed_batch, cql_weight=cql_w)` (line 258) — this
  time with the kwarg.

This entire pipeline existed in the codebase but was **never invoked**
from the CLI: `run_o2o.py` rolled its own (broken) inline replacement.

**Detected by.** Reading both files side-by-side after Bug 3 was fixed,
to figure out where the `--adaptive_conservatism` flag I had just added
would actually take effect. It didn't. The flag was a no-op until I
swapped the inline loop for `agent.finetune_online`.

**Fix.** Replace the inline loop (and add metrics + cql-trajectory save):
`scripts/run_o2o.py:307-356`. The new code calls `finetune_online`,
which is the only path that respects the `adaptive_conservatism` config
flag.

**Integrity impact — read this carefully.** Pre-fix, every `o2o` phase
run, regardless of CLI flags, was a "fully-online fine-tune from offline
init with zero conservatism" — i.e., it neither was the **adaptive O2O**
condition described in the proposal nor the **naive fine-tune** condition
in the four-way comparison nor anything in between. It was a third,
different condition that the experimental design had no name for.

Therefore: **the milestone's planned adaptive-O2O vs naive-fine-tune
comparison was silently a degenerate-case-vs-degenerate-case comparison
until this fix.** All Phase 2C results post-date this fix; any prior
"adaptive O2O" numbers from earlier branches must be relabeled or
discarded. We will state this explicitly in the limitations section of
the paper rather than retroactively rewrite the milestone.

---

## Cross-bug dependencies and CI implications

Bugs 2, 3, and 4 are stacked: fixing 2 is a prerequisite to even
encountering 3, and fixing 3 is a prerequisite to encountering 4. The
fact that all three lived in `run_o2o.py`'s `o2o` branch and were
discovered in sequence by a single five-minute smoke test indicates the
o2o phase had not been exercised end-to-end in the existing CI. We added
no new CI step for this phase (Phase 2 budget did not allow it), but
recommend a smoke-level integration test
(`tests/test_o2o_pipeline_smoke.py`) that runs `--phase=o2o
--n_offline_updates=10 --n_online_steps=20` as a follow-up before any
extension of this work.

The `MomentumPolicy` bug (1) is independent of the O2O cluster but has
the same flavor: a unit test for behavior-policy diversity would have
caught it in seconds.
