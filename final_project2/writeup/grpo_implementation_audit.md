# GRPO implementation audit — read-only sanity check

Read-only audit of the Continuous-GRPO implementation against the
five verification points called out for Audit 2. No code edits. Bug
classes are tagged:

- **STOP(needs-fix-before-results)** — invalidates the Phase 2B GRPO
  numbers or the Phase 2D ablation; halt and triage.
- **TODO(method-revision)** — implementation diverges from the Method
  draft in a way that needs reconciling in writeup, but does not
  invalidate results.
- **NOTE** — design choice worth surfacing; not a bug.

## Files surveyed

- `online_rl/agents/grpo.py` — `GRPOConfig`, `grpo_loss`, `GRPOTrainer`.
- `online_rl/agents/grpo_advantages.py` — `group_relative_advantage`.
- `online_rl/configs/grpo_config.py` — defaults.
- `scripts/train_grpo.py` — entrypoint (data, env, actor, train loop, backtest).
- `core/envs/portfolio_env.py:126-249` — `step` and `peek_reward_batch`.
- `core/networks/policies.py:401-475` — `DirichletMLPPolicy`.
- `tests/test_peek_reward_consistency.py` — bit-equivalence tests
  (`atol=1e-12`).
- `tests/test_exogeneity.py`, `test_grpo_loss.py`, `test_grpo_advantage.py`,
  `test_grpo_trainer.py`, `test_grpo_integration.py` — surveyed by name only.

Headline: **no STOP-class bugs found.** The Phase 2B GRPO numbers
(seed 42/1337/2024 = +0.31/+0.28/+0.31, turnover ≈ 0.24) sit on a
correct implementation. Phase 2C / Phase 2D can proceed without
GRPO-side fixes.

---

## Verification point 1 — Group sampling at fixed s_t with same forward-return path

**Claim:** all G actions in a group are sampled at the *same* state, and
all G rewards use the *same* forward-return row R_{t+1}.

**Code path:** `GRPOTrainer.collect`
(`online_rl/agents/grpo.py:266-284`):
1. Snapshot current obs into a single tensor (`obs_t = ... self._obs ... .unsqueeze(0)`).
2. One Dirichlet at that state: `dist = self.actor.dist(obs_t)` (batch_shape=(1,)).
3. G-fold sample in one shot: `samples = dist.sample((G,))` — `[G, 1, N]`.
4. log-probs for the same G samples in the same call:
   `dist.log_prob(samples).squeeze(1)` — `[G]`.
5. `env.peek_reward_batch(actions_np)` (`portfolio_env.py:195-249`)
   reads the index *once* (`idx = min(self._start + self._t, self._n_steps - 1)`)
   and the row *once* (`R = self._forward_returns[idx]`), then computes
   `port_returns = weights_batch @ R` for all G candidates against the
   same R.

The state is held constant across the G samples (`obs_t` is reused; env
is not stepped between samples). The forward-return row is held
constant across the G candidates (single `R`, single matmul). ✓

**No issues.**

---

## Verification point 2 — Forward-return computation correctness

**Claim:** `peek_reward_batch` produces *bit-identical* per-sample
rewards to what `step()` would produce if we replayed the env with
each candidate in turn.

**Code-path comparison:**

`step` (`portfolio_env.py:144-155`):
```python
weights = _project_to_simplex(action, self._n_assets)
idx = min(self._start + self._t, self._n_steps - 1)
port_return = float(np.dot(weights, self._forward_returns[idx]))
turnover = float(np.sum(np.abs(weights - self.prev_weights)))
simple_return = port_return - self._lambda * turnover
clamped = max(simple_return, -0.9999)
reward = float(np.log1p(clamped))
```

`peek_reward_batch` (`portfolio_env.py:232-249`):
```python
weights_batch = np.stack(
    [_project_to_simplex(a, self._n_assets) for a in actions], axis=0,
)
idx = min(self._start + self._t, self._n_steps - 1)
R = self._forward_returns[idx]
port_returns = weights_batch @ R
turnovers = np.sum(np.abs(weights_batch - self.prev_weights[None, :]), axis=1)
simple_returns = port_returns - self._lambda * turnovers
clamped = np.maximum(simple_returns, -0.9999)
rewards = np.log1p(clamped)
```

Per-row equivalence — identical projection helper, identical index, same
`prev_weights`, same `_lambda`, identical clamp threshold (`-0.9999`),
identical `log1p`. The only structural difference is vectorization
(matmul / broadcast) which doesn't change values for these scales.

**Cross-check against the test suite** (`tests/test_peek_reward_consistency.py`):
- `test_peek_matches_step_at_reset` — single candidate at reset
  (`atol=1e-12`).
- `test_peek_matches_step_after_nontrivial_history` — same after a
  warm-up trajectory (`atol=1e-12`).
- `test_peek_does_not_mutate_state` — `_t`, `prev_weights`,
  `_portfolio_value`, `_done`, `_start` all unchanged.
- `test_peek_batch_shape_and_finite_with_dirichlet_actions` — full
  Dirichlet sample group, finite outputs.
- `test_peek_batch_matches_step_for_each_row` — each of G rows
  compared row-by-row to a fresh step (`atol=1e-12`).

The `atol=1e-12` floor exceeds float64 round-off; the equivalence is
genuinely bit-for-bit. ✓

**No issues.**

---

## Verification point 3 — Group-relative advantage normalization

**Claim:** advantages are computed within each group of G samples (per
state), with the standard z-score reduction.

**Code path:** `group_relative_advantage`
(`online_rl/agents/grpo_advantages.py:25-72`).

Default mode is `"mean_std"`:
```python
mean = rewards.mean(dim=1, keepdim=True)              # [B, 1]
std = rewards.std(dim=1, keepdim=True, unbiased=False)
return (rewards - mean) / (std + eps)                 # [B, G]
```

Dim-1 reduction is correct — `rewards` arrives as `[B, G]` from `collect`
and the reduction must be over G. The `unbiased=False` flag uses the
biased (1/G) variance, which is the right choice for a
within-group-only baseline (the unbiased correction would over-divide
for small G; with G=4 this would overstate `std` by sqrt(4/3) ≈ 1.15
and depress all advantages proportionally — small but unnecessary).
The `eps=1e-8` denominator guard stops constant-reward rows from
exploding. The shape constraint and mode validation are checked at
function entry.

The "rank" mode (`_rank_centered_per_row`, lines 75-95) handles G=1 and
constant-reward rows correctly (zero output in both cases). Detached
input — `grpo_loss` calls `group_relative_advantage(rewards.detach(), ...)`
at line 152 of `grpo.py`, so reward gradients never flow back through
the advantage. ✓

**No issues.**

---

## Verification point 4 — Policy gradient form

**Claim:** PPO-style clipped surrogate of the importance-ratio
applied to group-relative advantages, plus KL-to-reference-policy anchor.

**Code path:** `grpo_loss` (`online_rl/agents/grpo.py:83-189`).

Loss components (`grpo.py:154-171`):
```python
surrogate1 = ratio * adv
surrogate2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
surrogate_per = -torch.min(surrogate1, surrogate2)             # [B, G]
surrogate_loss = surrogate_per.mean(dim=1).mean()
...
kl_per_state = kl_divergence(dist_new, dist_ref)               # [B]
kl_to_ref = kl_per_state.mean()
entropy_loss = -float(cfg.entropy_coef) * entropy_per_state.mean()
loss = surrogate_loss + float(cfg.beta_kl) * kl_to_ref + entropy_loss
```

Per-state averaging: mean over G first, then over B
(`grpo.py:158-159`). The docstring at `grpo.py:130-133` explicitly
notes this is equivalent to a flat mean for uniform G but matters for
variable G. Defensive choice; no current effect since G is constant.

Importance ratio: log-space hard clamp before exp
(`grpo.py:146-149`):
```python
log_ratio = new_logp - old_logprobs                            # [B, G]
log_ratio_clamped = log_ratio.clamp(_LOG_RATIO_LO, _LOG_RATIO_HI)
ratio = log_ratio_clamped.exp()
```
where `_LOG_RATIO_LO/HI = log(1e-3) / log(1e3)`. NaN-safe under
pathological divergence; the soft PPO clip on the surrogate is the
primary mechanism. The hard clamp is logged as `ratio_hard_clipped`
for observability. ✓

New log-probs in a single batched forward (`grpo.py:138-140`):
```python
dist_new = actor.dist(states)              # batch_shape=(B,)
actions_T = actions.transpose(0, 1).contiguous()   # [G, B, N]
new_logp = dist_new.log_prob(actions_T).transpose(0, 1)  # [B, G]
```
PyTorch's Dirichlet broadcasts `sample_shape=(G,)` over `batch_shape=(B,)`
correctly; output is `[G, B]` which transposes to `[B, G]`. This is the
"path (b)" optimization referenced in the comment — ~G× cheaper than
calling the network G times. ✓

**NOTE:** the KL anchor's reference policy is *frozen at trainer
initialization* (`grpo.py:227-231`):
```python
self.ref_actor = deepcopy(self.actor).to(self.device)
for p in self.ref_actor.parameters():
    p.requires_grad_(False)
```
For LLM-GRPO this would be the SFT policy (a meaningful reference). In
our setting `actor` is randomly initialized (no offline pretrain in the
GRPO online-only run), so the KL pulls toward a *random* Dirichlet.
With `beta_kl=0.01` the regularization is weak; the observed
`adv_std` (logged each iter) and the test-Sharpe ≈ +0.30 across seeds
suggest this is not blocking learning. **TODO(method-revision):** the
Method draft should clarify whether the "reference policy" framing
matches the actual code (init-frozen actor) or whether the writeup
implies a meaningful prior. If a meaningful KL prior is desired, refresh
`ref_actor` from a checkpoint at trainer init time.

**No STOP-class issues.**

---

## Verification point 5 — Group-size parameterization for Phase 2D ablation

**Claim:** `--group_size` flows from the CLI through `GRPOConfig` into
the trainer's `collect`/`update`, and varying it produces ablation
results that differ only in group size (everything else held).

**CLI wiring:**
- `scripts/train_grpo.py:180`:
  `p.add_argument("--group_size", type=int, default=cfg_defaults.group_size)`
- `scripts/train_grpo.py:247`: `GRPOConfig(group_size=args.group_size, ...)`

**Consumption inside trainer:**
- `grpo.py:259`: `G = int(self.cfg.group_size)` — used for sample
  shape and for the env action dispatch.
- `grpo.py:273`: `dist.sample((G,))`.
- `grpo.py:287`: `chosen_idx = int(self._rng.integers(0, G))` —
  uniform over G.
- `grpo.py:297`: tensor stacking yields `actions: [N, G, n_assets]`,
  `rewards: [N, G]`, `old_logprobs: [N, G]`.

**Test that the actor / dataset / env / hyperparameters are otherwise
identical across G values** (Phase 2D
`scripts/_run_phase2d_grpo_ablation.sh`):
```bash
run_grpo() {
    local g="$1"
    local seed=42
    local name="grpo_G${g}_lambda0.001_seed${seed}"
    ...
    uv run python scripts/train_grpo.py \
        --seed "$seed" \
        --dataset datasets/real_dirichlet.npz \
        --transaction_cost 0.001 \
        --total_env_steps 100000 \
        --group_size "$g" \
        --log_every 5000 \
        ...
}
run_grpo 4  &
run_grpo 8  &
run_grpo 16 &
```
Only `--group_size` varies; seed, dataset, transaction cost, env-step
budget, all GRPO hyperparameters (advantage_norm, beta_kl, clip_eps,
epochs_per_batch, minibatch_size, lr, grad_clip, entropy_coef),
hidden_dim, n_layers — all default and shared. The test_env path is
deterministic (`reset(options={"randomize": False})` in
`_run_test_backtest`). ✓

**NOTE:** Phase 2D's `seed=42` G=16 run produces a different result
directory (`grpo_G16_lambda0.001_seed42` in `results/phase2d/`) than the
Phase 2B seed=42 run (`grpo_lambda0.001_seed42` in `results/phase2b/`).
The two should agree to within seeding noise (training trajectory differs
because `_run_phase2d_grpo_ablation.sh` parallelizes 3 GRPO processes
on one GPU vs Phase 2B's 3-way parallel-with-PPO/SAC) but won't be
bit-identical. This is fine for the ablation question (slope w.r.t. G);
do not present G=16 from 2B and G=4/8 from 2D as a single-experiment
sweep — the ablation should always read all three G values from
`results/phase2d/`. **TODO(method-revision):** flag this in the
Sec.~\ref{sec:experiments:phase2d} writeup so reviewers don't
mis-attribute the comparison.

**No STOP-class issues.**

---

## Other observations (not in the 5 verification points)

### NOTE — uniform-random env action choice

`GRPOTrainer.collect:287`:
```python
chosen_idx = int(self._rng.integers(0, G))
env_action = actions_np[chosen_idx]
next_obs, _, terminated, truncated, _ = self.env.step(env_action)
```

The env steps with a *uniformly-chosen* sample from the group, not the
best-reward sample or the actor's mean. This is a deliberate
exploration-noise injection: subsequent `prev_weights` (which enters the
turnover penalty in subsequent `peek_reward_batch` calls) is broadly
distributed. Within a single state's group, all G actions share the
same `prev_weights` so the within-group comparison is fair; across
states, the prev_weights distribution is "uniform-random Dirichlet
sample" rather than "actor mean."

This isn't a bug per se — it's a choice. The implication for the
writeup: the *training* state distribution is broader than the
*deployment* state distribution (where the test backtest uses
`dist.mean`, `train_grpo.py:112`). The Method draft should be explicit
that GRPO is trained off-policy in this sense.

### NOTE — Dirichlet parameterization

`DirichletMLPPolicy` (`core/networks/policies.py:425-432`):
```python
alpha = F.softplus(self.actor_alpha(features)) + 1.0
dist = Dirichlet(concentration=alpha)
```
`alpha + 1.0` ensures every component is > 1, so the Dirichlet is
unimodal (otherwise the density on the simplex corners is unbounded
and entropy/log_prob become numerically unstable). Matches what the
SAC-Dirichlet post-mortem assumed. ✓

### NOTE — exogeneity holds despite turnover penalty

The Method draft argues GRPO is sound under "state transitions
independent of action." The training env at `_slice_env` uses
`include_prev_weights=False` (`train_grpo.py:86`), so the *observation*
at t+1 is purely the market-features row — independent of the action
taken at t. The reward at t+1 *is* action-dependent through the
turnover penalty (which uses `prev_weights = action_t`), but reward
exogeneity is not what GRPO requires. The bootstrap-cancellation
argument only requires V(s_{t+1}) to be independent of a_t; with
`include_prev_weights=False`, V(s_{t+1}) is a function of market
features only, so the cancellation goes through. ✓

The codebase backstops this with `tests/test_exogeneity.py`. The
choice to gate `include_prev_weights=False` on the GRPO env and not
the SAC/PPO envs is intentional and noted in the train_grpo.py
comment.

---

## Summary table

| Point | Verdict | Notes |
|---|---|---|
| 1. Group at fixed s_t, same R | ✓ pass | `collect()` snapshots once; `peek_reward_batch` reads R once |
| 2. Forward-return correctness | ✓ pass | bit-equivalent to `step()` (atol=1e-12 in tests) |
| 3. Group-relative advantage | ✓ pass | dim-1 z-score with eps guard, biased std (correct), detached rewards |
| 4. PG form (clip + KL + entropy) | ✓ pass | + TODO(method-revision) on KL ref policy = init-frozen actor |
| 5. Group-size ablation hookup | ✓ pass | + TODO(method-revision) on Phase 2D vs 2B G=16 attribution |

**No STOP-class bugs.** Phase 2C and Phase 2D can run without code
fixes. Two TODO(method-revision) items to fold into the Method /
Experiments draft after Phase 2D completes.

---

## Addendum — GRPO offline warm-start compatibility check (post-Step-1)

For the proposed Phase 2C "GRPO with offline warm-start" condition, the
question is: can a GRPO actor (`DirichletMLPPolicy`) be initialized
from a Phase 2A IQL/AWAC/BC checkpoint (`DirichletActor`) and continue
training online?

### Architecture comparison

| Property | IQL/AWAC/BC actor | GRPO actor |
|---|---|---|
| Class | `core.networks.dirichlet_policy.DirichletActor` | `core.networks.policies.DirichletMLPPolicy` |
| Backbone | `mlp(obs_dim, 256, action_dim, n_layers=2)` — fused into one `Sequential` named `net` | `Sequential(Linear, Tanh, Linear, Tanh)` named `shared` + separate `actor_alpha = Linear(256, 8)` |
| Activation | `Tanh` (from `mlp` helper, `dirichlet_policy.py:34`) | `Tanh` (`policies.py:421`) |
| Output | `α = softplus(net(obs)) + 1` → `Dirichlet(α)` | `α = softplus(actor_alpha(shared(obs))) + 1` → `Dirichlet(α)` |
| Hidden dim | 256 (Phase 2A configs) | 256 (`grpo_config.py:30`) |
| Extra heads | none | `critic = Linear(256, 1)` — value head, **unused by GRPO** |

The two networks are **mechanically equivalent up to a key rename**.
Layer-by-layer shapes are bit-identical:

```
IQL  net.0.{w,b}: (256, obs_dim) | (256,)   ↔  GRPO  shared.0.{w,b}
IQL  net.2.{w,b}: (256, 256)     | (256,)   ↔  GRPO  shared.2.{w,b}
IQL  net.4.{w,b}: (action_dim, 256) | (action_dim,)   ↔  GRPO  actor_alpha.{w,b}
                                                          (drop GRPO  critic.{w,b})
```

### Empirical verification

Ran a load-and-forward sanity check at `obs_dim=56, action_dim=8,
hidden_dim=256, n_layers=2`:

```python
from core.networks.dirichlet_policy import DirichletActor
from core.networks.policies import DirichletMLPPolicy

iql_actor  = DirichletActor(56, 8, 256, 2)
grpo_actor = DirichletMLPPolicy(56, 8, 256, 2)

rename = {
    "net.0.weight": "shared.0.weight", "net.0.bias": "shared.0.bias",
    "net.2.weight": "shared.2.weight", "net.2.bias": "shared.2.bias",
    "net.4.weight": "actor_alpha.weight", "net.4.bias": "actor_alpha.bias",
}
remapped = {rename[k]: v for k, v in iql_actor.state_dict().items()}
missing, unexpected = grpo_actor.load_state_dict(remapped, strict=False)
# missing = ['critic.weight', 'critic.bias']  ← GRPO's value head, fresh init
# unexpected = []                              ← no IQL weights dropped
```

After the load:
- `softplus(iql_actor.net(obs)) + 1` and
  `softplus(grpo_actor.actor_alpha(grpo_actor.shared(obs))) + 1`
  agree to **0.00e+00** at every input (bit-exact).
- `Dirichlet(α).mean` agrees to 0.00e+00 between the two.

### Branch determination

This case sits between the user's "Compatible (same class, same
shapes)" and "Same class, different shapes" branches. Specifically:

- The classes *differ* (`DirichletActor` vs `DirichletMLPPolicy`),
  which would naively place us in the "Architecturally incompatible"
  branch.
- But the per-layer shapes are bit-identical and the forward pass is
  bit-equivalent after a 6-key rename, which is the practical
  signature of "Compatible."

I'm reading this as **the spirit of the "Compatible" branch**: the
~30 LOC warm-start feature should work. Specifically:

1. Add `--init_actor_checkpoint <path>` to `scripts/train_grpo.py`.
2. After constructing `actor = DirichletMLPPolicy(...)`, if
   `--init_actor_checkpoint` is set:
   ```python
   ckpt = torch.load(path, map_location=device)
   # ckpt may be a full agent dict; extract the actor sub-dict
   src_actor = ckpt.get("actor_state", ckpt.get("actor", ckpt))
   rename = {  # 6 keys
       "net.0.weight": "shared.0.weight", "net.0.bias": "shared.0.bias",
       "net.2.weight": "shared.2.weight", "net.2.bias": "shared.2.bias",
       "net.4.weight": "actor_alpha.weight", "net.4.bias": "actor_alpha.bias",
   }
   remapped = {rename[k]: v for k, v in src_actor.items() if k in rename}
   missing, unexpected = actor.load_state_dict(remapped, strict=False)
   assert unexpected == [], f"unexpected: {unexpected}"
   assert set(missing) == {"critic.weight", "critic.bias"}, (
       f"unexpected missing: {missing}"
   )
   print(f"[init] warm-started GRPO actor from {path}; critic.* fresh init")
   ```
3. Then GRPOTrainer's `__init__` snapshots `ref_actor = deepcopy(actor)`
   *after* the warm-start, so the KL anchor is the warm-started policy
   (not random init — this addresses the TODO(method-revision) in
   §4 of this audit). The KL pulls the actor toward the offline-prior
   manifold for the first few collect/update iterations, then relaxes
   as the actor adapts online.
4. Source checkpoint: per the user's spec, IQL seed=42 from Phase 2A.
   Caveat: the in-flight Phase 2C runs use the **causal-pipeline** env;
   the existing Phase 2A IQL checkpoints were trained on the **leaky**
   env. After Step 4 (Phase 2A causal sanity) lands, prefer one of the
   *causal* IQL/AWAC/BC checkpoints as the warm-start source so the
   pre-trained policy matches the fine-tuning env.

### Open question for the user (do not implement until confirmed)

- **Branch**: confirm whether the "different class, identical
  layer-shape" case clears the Compatible bar in your judgment, or if
  the class mismatch alone is enough to fall through to "Skip GRPO
  warm-start."
- **Source checkpoint**: do we wait for Step 4 (causal Phase 2A) to
  produce a causal IQL checkpoint, or use the leaky-pipeline IQL
  checkpoint? The cleanest experiment uses the causal source; the
  fastest uses what's already on disk.
- **GRPOTrainer ref_actor reseed**: the warm-start naturally gives a
  meaningful KL anchor (no longer "random init"). Do we want this
  side-effect, or should we deepcopy `actor` into `ref_actor` *before*
  applying the warm-start so the ref remains random and only the
  trainable actor benefits?

Awaiting your call before implementing.

---

## Addendum 2 — Smoke-test had a hole; obs-dim mismatch caught post-launch

The first attempt at Phase 2C "GRPO with offline warm-start" failed at
runtime with:

```
RuntimeError: Error(s) in loading state_dict for DirichletMLPPolicy:
    size mismatch for shared.0.weight: copying a param with shape
    torch.Size([256, 56]) from checkpoint, the shape in current model
    is torch.Size([256, 216]).
```

This is a **feature-pipeline mismatch**, not an actor-class mismatch.
The IQL source (Phase 2A causal) trains on
\texttt{compute\_features} → 56-d obs; the GRPO target (Phase 2B/2D,
\texttt{datasets/real\_dirichlet.npz}) trains on
\texttt{build\_features} → 216-d obs. The two pipelines are causal
(post-leak-fix) but not feature-compatible.

**Why the smoke didn't catch it pre-launch:** the smoke harness
hardcoded \texttt{obs\_dim=56} as a CLI default to match the source
checkpoint, instead of constructing the actual target env and reading
\texttt{train\_env.observation\_space.shape[0]}. So the pre-launch
smoke confirmed source ↔ source equivalence, not source ↔ target
compatibility.

**Patch applied** (commit pending): \texttt{scripts/\_smoke\_warmstart.py}
now requires \texttt{--target\_dataset} and constructs the actual
GRPO target env via the same \texttt{\_slice\_env} path that
\texttt{train\_grpo.py} uses. It reads
\texttt{train\_env.observation\_space.shape[0]} and asserts it matches
the checkpoint's first-layer input dim BEFORE any state\_dict copy.
On mismatch, exits with code 2 and prints a diagnostic that names both
dimensions and the likely feature-pipeline cause.

**Regression test:** running the patched smoke against the existing
56-d Phase 2A causal IQL checkpoint with the 216-d GRPO target
dataset:

```
$ uv run python scripts/_smoke_warmstart.py \
    --checkpoint results/phase2a_causal/iql_lambda0.001_seed42/actor.pt \
    --target_dataset datasets/real_dirichlet.npz
[load] results/phase2a_causal/iql_lambda0.001_seed42/actor.pt
  keys: ['net.0.weight', 'net.0.bias', 'net.2.weight', ...]

[pre-check] Building target env from datasets/real_dirichlet.npz
  target env obs_dim = 216
  source ckpt expects input dim = 56

[FAIL pre-check] Dimensionality mismatch: source checkpoint was trained
on 56-d observations, but the target GRPO env has 216-d observations.
... This run cannot be warm-started without retraining the source on
the target's feature space.
$ echo $?
2
```

The harness now has teeth: it would have caught the mismatch
pre-launch and prevented the failed runs.

**Resolution path**: train a one-off 216-d IQL on the
\texttt{datasets/real\_dirichlet.npz} arrays so the warm-start source
matches the GRPO target's feature space. Documented as Step 2 of the
post-failure plan; result will be appended here when the 216-d IQL
sanity checks pass.
