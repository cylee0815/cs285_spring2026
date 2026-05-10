# SAC degeneracy post-mortem — root cause: unreachable target entropy

## The observed pathology

Phase 2B Batch 1 results (3 SAC-Dirichlet seeds, λ=0.001, 100k env
steps) at evaluation time:

| seed | test_sharpe | cum_return | max_dd  | turnover |
|------|-------------|------------|---------|----------|
| 42   | +0.9413     | +0.3255    | 0.1188  | 0.00000  |
| 1337 | +0.9413     | +0.3255    | 0.1188  | 0.00000  |
| 2024 | +0.9413     | +0.3255    | 0.1188  | 0.00000  |

All three seeds report identical metrics to four decimals; turnover is
exactly zero. Equal-weight test Sharpe on the same window is 0.9531
(`runs/ablation/.../metrics.json:ew_sharpe`). The deterministic agent
that always emits `[1/8, ..., 1/8]` would score essentially these
numbers — **the policy has collapsed to literal equal-weight, with no
re-allocation across the 1,061-step test window.** Training-time logs
show the auto-tuned temperature τ growing from 18 at step 10k to
2.4×10¹⁰ at step 80k while the entropy floor sits at exactly −8.525 from
step 10k onward (the Dir(1, …, 1) entropy on the 7-simplex,
−log(7!) ≈ −8.525).

The collapse is not a seed artifact (the seeds did diverge slightly
during training: seed 1337 reached val_sharpe 1.186 vs. 1.295 for the
others at step 80k, before all three converged to the same final
state). It is a method-level pathology that any seed reaches.

## Root cause: target entropy is unreachable, in the wrong sign

`online_rl/agents/sac_dirichlet.py:88`:
```python
self.target_entropy = np.log(action_dim)
```
With `action_dim = 8`, `target_entropy = log 8 ≈ +2.079`.

This is **two bugs in one line**:

### Bug A — wrong sign convention

The standard SAC temperature target convention (Haarnoja et al. 2018,
SB3, CleanRL, the original `softlearning` implementation) is
`target_entropy = -action_dim` for diagonal-Gaussian actions. The sign
is *negative*. Setting `target_entropy = +log(action_dim)` flips it.

The temperature loss at
`online_rl/agents/sac_dirichlet.py:260-261`:
```python
# τ loss: -τ * (log π + H_target)
temp_loss = -(self.log_temperature * (log_prob + self.target_entropy).detach()).mean()
```
Differentiating w.r.t. `log_α`:
```
∂L/∂log_α = -(log π + H_target).
```
The minimizer therefore drives `log_α` upward whenever
`(log π + H_target) > 0` and downward when it is negative. With an
*observed* `log_prob ≈ +8.525` (Dirichlet density at typical samples is
above 1, so log-density is positive), and `target_entropy = +2.08`, the
sum is **always positive (≈ +10.6)** and `log_α` increases without
bound. After 100k steps with Adam at lr = 3e-4, that produces the
observed τ ≈ 10¹⁰.

The correct sign convention puts `H_target` **negative** so the loss
saturates the temperature near a finite value at convergence (entropy
matches the target). Here `H_target = +log(8)` was likely copy-pasted
from a discrete-action SAC implementation where the Boltzmann actor
has entropy in `[0, log(N)]`; that range does not apply to the
continuous Dirichlet on the (N-1)-simplex.

### Bug B — wrong magnitude even with the sign fixed

For a continuous Dirichlet on the (N-1)-simplex:
- The maximum entropy is achieved by `Dir(1, 1, …, 1)` and equals
  `H_max = -log((N-1)!)`. For N=8, `H_max = -log(7!) ≈ -8.525`.
- The minimum entropy is `-∞` (concentrated Dirichlets have
  arbitrarily negative entropy).

So the achievable range of `H` is `(-∞, -8.525]` — a half-line of
**strictly negative** values. The Gaussian-SAC convention
`target_entropy = -action_dim = -8` lies just above the maximum, so it
is *unreachable from above*; the temperature is asked to push entropy
above its maximum, which it cannot, and the temperature update keeps
demanding more concentration → α decreases → entropy regularization
collapses → Q-only objective → policy concentrates on whatever the
critic likes most.

A correct `target_entropy` for an N-simplex Dirichlet should sit
**inside** `(-∞, -log((N-1)!))`. A common choice is a fraction of the
maximum, e.g.:
```python
self.target_entropy = 0.9 * (-np.log(math.factorial(action_dim - 1)))
                      # ≈ 0.9 * (-8.525) = -7.673
```
or the heuristic from the discrete-Dirichlet adaptation literature,
`H_target = -(N-1)` (= -7 for N=8), which gives a slightly more
exploratory floor.

### Bug C (related, but not the proximate cause) — temperature is
unbounded

`self.log_temperature = nn.Parameter(torch.zeros(1, device=device))`
(`online_rl/agents/sac_dirichlet.py:89`) is a free parameter; nothing
in the optimizer or its forward use clamps `temperature` to a sane
range. Many production SAC implementations clamp τ ∈ `[1e-3, 1e2]` (or
clamp the gradient of `log_α`) precisely to prevent runaway when the
target is mis-set. Adding such a clamp would have masked Bug A; it
would not have *fixed* the underlying convention error.

## Sanity check — actor loss sign is correct

`online_rl/agents/sac_dirichlet.py:226-227`:
```python
# Actor loss: minimize (τ * log_prob - Q) = maximize (Q - τ * entropy)
actor_loss = (self.temperature.detach() * log_prob - q_val).mean()
```
This is the standard SAC actor loss: minimize `α·log π(a|s) - Q(s,a)`.
The actor pushes log π *down* (i.e., makes the policy more entropic)
to the extent that α is large, and chases Q to the extent that α is
small. With our pathological τ ≈ 10¹⁰, the entropy term dominates by
ten orders of magnitude over any Q-signal: the actor's only effective
objective is "maximize entropy" → uniform Dir(1, …, 1) → identical
equal-weight allocations every step → turnover ≡ 0.

Conclusion: **Bug A is the proximate cause, Bug B explains why the
sign-corrected version would still misbehave, Bug C is the missing
guardrail.** The actor loss is otherwise correct.

## What the correct implementation should look like

A non-patch sketch (do not apply to the repo mid-run):
```python
# online_rl/agents/sac_dirichlet.py:88
import math
# Continuous Dirichlet on (N-1)-simplex: max entropy is -log((N-1)!).
# Target sits inside the achievable range; the 0.9 factor is a small
# margin so the temperature update can still push entropy upward.
H_max = -math.log(math.factorial(action_dim - 1))
self.target_entropy = 0.9 * H_max     # e.g. -7.67 for N=8

# online_rl/agents/sac_dirichlet.py:89
# Optional but strongly recommended: clamp the log-temperature so
# numerical runaway under any future config error is bounded.
self._log_temp_clamp = (math.log(1e-3), math.log(1e2))

@property
def temperature(self):
    return self.log_temperature.clamp(*self._log_temp_clamp).exp()
```

(No code change to actor loss or temperature loss is needed; the sign
convention is fixed by changing only `target_entropy` and bounding
`log_temperature`.)

## Report-ready writeup (1 paragraph, ready to paste)

> SAC-Dirichlet, trained from scratch on the 2022Q1–2026Q1 test window
> at λ = 0.001 with 100,000 environment steps, collapses to a
> deterministic equal-weight allocation across all three seeds (test
> Sharpe 0.941 ± 0.000, turnover ≡ 0). The cause is a sign-and-magnitude
> error in the entropy-temperature target: the implementation sets
> `target_entropy = +log N` (a Gaussian-discrete-action heuristic), but
> the achievable entropy range for a continuous Dirichlet on the
> (N−1)-simplex is `(-∞, -log((N−1)!)]` — a half-line of strictly
> negative values. The unreachable positive target drives the
> auto-tuned temperature τ to ~10¹⁰ over training, after which the
> entropy regularizer dominates the Q-pursuit term by ten orders of
> magnitude and the actor optimizes for maximum entropy alone (uniform
> Dirichlet, equal-weight allocation, zero turnover). This is a
> known-class SAC-on-simplex failure mode. It motivates GRPO's
> entropy-free design: by replacing the temperature-regularized SAC
> objective with within-group reward standardization
> (Sec.~\ref{sec:method:grpo}), the simplex-action GRPO formulation
> sidesteps the auto-tune-on-bounded-entropy pathology entirely. We
> report the SAC test numbers as a method-level pathology rather than a
> baseline; a one-line target-entropy fix would likely recover
> SAC-Dirichlet to a comparable level, but is left as post-paper work.

## Decision needed (yours, not mine)

Whether to re-run Phase 2B SAC after Phase 2D completes with the fix
applied is your call. Arguments either way:

- **Re-run**: a corrected SAC-Dirichlet baseline strengthens the
  online-only condition in the four-way comparison; the current SAC
  rows essentially measure equal-weight, which trivializes
  Sec.~\ref{sec:experiments:phase2b}.
- **Don't re-run**: SAC's pathology is itself a finding that motivates
  GRPO; the report can present it as such. PPO-LSTM and GRPO are
  unaffected and provide the substantive online-only comparison. Phase
  2B SAC then plays the role of "naïve SAC-Dirichlet on simplex without
  per-distribution tuning" — a known pitfall, illustrated.

I do not recommend re-running mid-experiment. If a re-run happens, it
should be after Phase 2D finishes so the GRPO ablation is unaffected.
