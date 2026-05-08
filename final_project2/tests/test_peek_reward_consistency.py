"""Bit-equivalence guard between ``PortfolioEnv.peek_reward_batch`` and ``step()``.

GRPO computes per-state rewards for G candidate actions via
``peek_reward_batch``. The headline GRPO-vs-PPO comparison is fair only if
that batched scoring is numerically identical to the reward signal that
``step()`` would produce for the same ``(state, action)`` pair. This file
guards against silent formula drift (different clamp, missing log1p,
different prev_weights handling, etc.) that would contaminate the headline
result.

Strategy: instead of deep-copying the env (which is fragile across
gymnasium versions), we drive **two identically-seeded envs** through the
same prefix actions, then compare ``step(a)`` on env_1 with
``peek_reward_batch(a[None,:])[0]`` on env_2 (still in the pre-step state).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.envs.portfolio_env import PortfolioEnv


N_ASSETS = 8
N_FEATURES = 12
T = 50
EPISODE_LENGTH = 30
TC_LAMBDA = 0.001


def _make_env(data_seed: int = 0) -> PortfolioEnv:
    """Synthetic env — deterministic across runs for the given seed."""
    rng = np.random.default_rng(data_seed)
    features = rng.standard_normal((T, N_FEATURES)).astype(np.float32)
    returns = (rng.standard_normal((T, N_ASSETS)) * 0.015 + 0.0002).astype(np.float32)
    return PortfolioEnv(
        features=features,
        forward_returns=returns,
        transaction_cost_lambda=TC_LAMBDA,
        episode_length=EPISODE_LENGTH,
    )


def _reset_and_advance(env: PortfolioEnv, seed: int, prefix_actions: list[np.ndarray]):
    """Reset env and step through the given prefix to reach a non-trivial state."""
    env.reset(seed=seed)
    for a in prefix_actions:
        env.step(a)


def test_peek_matches_step_at_reset():
    """Right after reset, prev_weights = uniform — peek must equal step."""
    env_step = _make_env(data_seed=7)
    env_peek = _make_env(data_seed=7)
    env_step.reset(seed=123)
    env_peek.reset(seed=123)

    candidate = np.array([0.4, 0.0, 0.2, 0.0, 0.1, 0.0, 0.3, 0.0], dtype=np.float32)

    # peek BEFORE the step on env_step, so both envs are in identical state.
    peek_reward = env_peek.peek_reward_batch(candidate[None, :])
    _, step_reward, _, _, _ = env_step.step(candidate)

    assert peek_reward.shape == (1,)
    np.testing.assert_allclose(peek_reward[0], step_reward, atol=1e-12, rtol=0.0)


def test_peek_matches_step_after_nontrivial_history():
    """After several steps, prev_weights is non-uniform — still must match.

    This is the case GRPO actually exercises during training.
    """
    rng = np.random.default_rng(0)
    prefix = [rng.dirichlet(np.ones(N_ASSETS)).astype(np.float32) for _ in range(5)]

    env_step = _make_env(data_seed=11)
    env_peek = _make_env(data_seed=11)
    _reset_and_advance(env_step, seed=99, prefix_actions=prefix)
    _reset_and_advance(env_peek, seed=99, prefix_actions=prefix)

    # Sanity: both envs should now have identical prev_weights and step counter.
    np.testing.assert_allclose(env_step.prev_weights, env_peek.prev_weights, atol=1e-15)
    assert env_step._t == env_peek._t

    candidate = np.array([0.05, 0.05, 0.4, 0.05, 0.05, 0.3, 0.05, 0.05], dtype=np.float32)

    peek_reward = env_peek.peek_reward_batch(candidate[None, :])
    _, step_reward, _, _, _ = env_step.step(candidate)

    np.testing.assert_allclose(peek_reward[0], step_reward, atol=1e-12, rtol=0.0)


def test_peek_does_not_mutate_state():
    """Peeking must not advance _t, change prev_weights, or set _done."""
    env = _make_env(data_seed=3)
    env.reset(seed=5)
    # Run a few steps so prev_weights is non-uniform.
    for _ in range(4):
        env.step(np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float32))

    snapshot_t = env._t
    snapshot_prev = env.prev_weights.copy()
    snapshot_done = env._done
    snapshot_pv = env._portfolio_value
    snapshot_start = env._start

    # Score a chunky batch, multiple times.
    rng = np.random.default_rng(0)
    batch = rng.dirichlet(np.ones(N_ASSETS), size=8).astype(np.float32)
    for _ in range(3):
        _ = env.peek_reward_batch(batch)

    assert env._t == snapshot_t
    np.testing.assert_array_equal(env.prev_weights, snapshot_prev)
    assert env._done == snapshot_done
    assert env._portfolio_value == snapshot_pv
    assert env._start == snapshot_start


def test_peek_batch_shape_and_finite_with_dirichlet_actions():
    """G=8 simplex-valid actions — shape (8,), all finite, sane mean."""
    G = 8
    env = _make_env(data_seed=2)
    env.reset(seed=17)
    rng = np.random.default_rng(42)
    actions = rng.dirichlet(np.ones(N_ASSETS), size=G).astype(np.float32)

    rewards = env.peek_reward_batch(actions)

    assert rewards.shape == (G,)
    assert np.all(np.isfinite(rewards)), f"Non-finite rewards: {rewards}"
    # Daily logged returns sit ~ O(1e-3); mean should be of that order.
    assert abs(rewards.mean()) < 0.1, f"Mean reward unexpectedly large: {rewards.mean()}"


def test_peek_batch_matches_step_for_each_row():
    """Stronger: every row of a G-batch matches the corresponding step()."""
    G = 4
    rng = np.random.default_rng(0)
    candidates = rng.dirichlet(np.ones(N_ASSETS), size=G).astype(np.float32)

    # Drive a single env, peek the full batch, then per-row replay step()
    # on a freshly-seeded env to recover the per-row reward.
    env_peek = _make_env(data_seed=4)
    env_peek.reset(seed=21)
    # Advance to non-trivial prev_weights.
    for _ in range(3):
        env_peek.step(np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float32))
    peek_rewards = env_peek.peek_reward_batch(candidates)
    assert peek_rewards.shape == (G,)

    for g in range(G):
        env_step = _make_env(data_seed=4)
        env_step.reset(seed=21)
        for _ in range(3):
            env_step.step(np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float32))
        _, step_reward, _, _, _ = env_step.step(candidates[g])
        np.testing.assert_allclose(
            peek_rewards[g], step_reward, atol=1e-12, rtol=0.0,
            err_msg=f"row {g} disagreed: peek={peek_rewards[g]} step={step_reward}",
        )


def test_peek_rejects_wrong_shape():
    """Defensive check on input shape."""
    env = _make_env(data_seed=0)
    env.reset(seed=0)

    with pytest.raises(ValueError):
        env.peek_reward_batch(np.zeros(N_ASSETS, dtype=np.float32))  # 1-D
    with pytest.raises(ValueError):
        env.peek_reward_batch(np.zeros((4, N_ASSETS - 1), dtype=np.float32))  # wrong N


def test_peek_rejects_done_episode():
    """Cannot peek after episode end — same contract as step()."""
    env = _make_env(data_seed=0)
    env.reset(seed=0)
    a = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float32)
    while not env._done:
        env.step(a)
    with pytest.raises(RuntimeError):
        env.peek_reward_batch(a[None, :])
