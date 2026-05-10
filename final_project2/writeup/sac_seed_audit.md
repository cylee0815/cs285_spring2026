# SAC seed-threading audit

## Symptom

Phase 2B Batch 1 (SAC-Dirichlet × 3 seeds) at step 80,000 reported:

| seed | val_sharpe | tau              |
|------|-----------:|-----------------:|
| 42   | 1.295      | 24,345,184,256   |
| 1337 | 1.186      | 24,345,878,528   |
| 2024 | 1.295      | 24,339,703,808   |

Two of three seeds produce a val_sharpe inside 0.001 of each other; all
three taus agree to 0.03%. The trajectories diverge slightly but
converge to the same pathological attractor (tau → ∞, policy → uniform).

Two questions: (a) is the seed propagated correctly through every
stochastic component, and (b) does it matter — i.e., is the apparent
collapse an artifact of broken seeding, or a genuine SAC failure mode?

## Where seed is and is NOT propagated

Tracing every stochastic source in `online_rl/agents/sac_dirichlet.py`
under the call path `scripts/run_online_baselines.py`:

| stochastic source | seeded by `args.seed`? | evidence |
|---|---|---|
| Python `random` module | ✅ | `set_seed(seed)` calls `random.seed(seed)` (`utils/seed.py:50`). |
| NumPy global RNG | ✅ | `np.random.seed(seed)` (`utils/seed.py:51`). |
| Torch CPU + CUDA RNGs | ✅ | `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` (`utils/seed.py:62-64`). |
| Torch model weight init (actor, critic, target_critic, regime_encoder) | ✅ (via torch global RNG, set above before the SAC `__init__` runs) | `set_seed(args.seed)` is at `scripts/run_online_baselines.py:352`, before any model construction. |
| Torch `dist.sample()` for online action sampling | ✅ (uses torch global generator) | `Dirichlet(α).sample()` follows `torch.manual_seed`. |
| **Env `_rng` for random episode starts** | ❌ | `core/envs/portfolio_env.py:87`: `self._rng = np.random.default_rng()` — note **no seed argument**. `numpy.random.default_rng()` without a seed reads system entropy and **ignores** the numpy global RNG seeded by `np.random.seed()`. |
| **Env reset call sites in SAC** | ❌ | `online_rl/agents/sac_dirichlet.py:109` and `:156` call `env.reset()` with neither `seed=` nor `options=`. Every episode boundary picks a random start from the unseeded `env._rng`. |
| Replay buffer sample indices | ✅ (torch global generator) | `torch.randint` in `core/buffers/replay_buffer.py`. |
| cuDNN nondeterminism | warned-only, not bit-exact | `utils/seed.py:75-76`: `torch.use_deterministic_algorithms(True, warn_only=True)`. |

## Diagnosis

Seed propagation is **partially broken**. The torch-side randomness
(actor / critic init, action sampling, replay-buffer indexing) is
correctly seeded — `args.seed` does control those. But the
**environment's random episode starts** are seeded by system entropy at
`PortfolioEnv.__init__` time and stay un-touched by `set_seed`. Across
the three SAC processes, env episode starts therefore differ
*per process* (different start time, different fork ID) rather than
*per seed*. Within a process, episode starts are repeatable but the
seed has no effect on which starts are visited.

The "two seeds report identical val_sharpe" pattern is consistent with
this picture: the torch-side RNG produces tiny weight differences that
get washed out by the SAC tau auto-tune blowup. The env-side
randomness, which would normally drive divergent exploration, is
process-specific noise — uncorrelated with the seed argument.

A second-order issue: even if the env reset took a seed, SAC always
calls `env.reset()` with no kwargs, so the only opportunity to seed it
is at construction (which the runner does not do).

## Why this is not the root cause of the SAC pathology

Even with perfect seed threading, SAC's tau auto-tune would still
explode. The runaway is mathematical: the temperature loss
$-\tau (\log \pi(w \mid s, h) + H_{\text{target}})$ has positive
gradient when entropy is below target, and the actor is fighting that
gradient by emitting concentrated Dirichlets. With $H_{\text{target}}
= \log 8 \approx 2.08$ and observed entropy $-8.5$, the gap is
~10.6 nats, multiplied by tau, summed across the buffer, integrated
over 100k steps. That's a sufficient explanation for tau → 1e10
without invoking seed mistakes. We will document the SAC-Dirichlet
collapse as a method-level finding, not a seeding artifact.

## Proposed fix (one paragraph, do not apply mid-run)

Add a `seed` parameter to `SACDirichlet.__init__`, call
`self.env.reset(seed=seed)` at line 109 to fix the initial reset,
maintain a `self._seed_seq` (`numpy.random.SeedSequence(seed)`) and
use `self._seed_seq.spawn(1)[0].generate_state(1)[0]` at every
subsequent `env.reset(...)` to give it a deterministic per-episode
seed, and thread `args.seed` through from `scripts/run_online_baselines.py`
on construction. The same fix applies at
`hybrid_rl/agents/o2o_agent.py:84` (where SACDirichlet is constructed
inside O2OAgent). Estimated diff: ~12 lines, two files. Validation: a
unit test that constructs two `SACDirichlet` instances with the same
seed and asserts identical `_obs` after the first `collect_step` ---
which is currently impossible given the unseeded env. Do not apply
until Phase 2B finishes; mid-run state changes invalidate the in-flight
metrics.json results.

## Reporting note

For the Phase 2 paper: report the three Phase 2B SAC seeds as
"per-process variance, env-randomness uncontrolled" rather than as
"per-seed variance". In the limitations section, note that
seed-threading on the env side will be addressed in the post-paper
follow-up; the SAC collapse story does not depend on the seeding fix
because the failure mode is independently characterized by the tau
trajectory.
