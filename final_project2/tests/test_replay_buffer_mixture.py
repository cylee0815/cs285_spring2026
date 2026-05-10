"""Tests for `ReplayBuffer.load_from_env` policy-mixture support.

Covers:
  1. Backward compat — no policy arg -> uniform Dirichlet behavior unchanged.
  2. Mixture sampling proportions match weights to within tolerance.
  3. Per-episode policy is held constant within an episode (changes only on
     env.reset()).
  4. Mutual exclusivity: passing both `policy` and `policy_mixture` raises.
  5. Validation of malformed mixture entries.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.spaces import Box

from core.buffers.replay_buffer import ReplayBuffer


class _StubEnv(gym.Env):
    """Minimal env that terminates every `episode_length` steps.

    Exposes deterministic obs/reward; does NOT report `executed_weights` so
    the buffer stores the policy-emitted action verbatim — letting the test
    fingerprint which policy produced each transition.
    """

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 4, action_dim: int = 3, episode_length: int = 5):
        super().__init__()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(action_dim,))
        self.episode_length = episode_length
        self._t = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.full(self.observation_space.shape, self._t, dtype=np.float32)
        terminated = self._t >= self.episode_length
        return obs, 0.0, bool(terminated), False, {}


def _make_const_policy(value: float, action_dim: int):
    """Returns a policy that emits a constant simplex vector with one hot-ish
    coordinate. Used as a fingerprint."""
    base = np.full(action_dim, (1.0 - value) / (action_dim - 1), dtype=np.float32)

    def _pol(_obs):
        out = base.copy()
        out[0] = value
        return out

    return _pol


@pytest.fixture
def buffer_factory():
    def _make(capacity=200, obs_dim=4, action_dim=3):
        return ReplayBuffer(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=torch.device("cpu"),
        )

    return _make


def test_default_policy_uniform_dirichlet(buffer_factory):
    """No policy arg -> uniform Dirichlet(1) sampling. Sanity-check that
    actions are valid simplex vectors with mean ~ 1/n.
    """
    env = _StubEnv(action_dim=8, episode_length=10)
    buf = buffer_factory(capacity=500, action_dim=8)
    np.random.seed(0)
    buf.load_from_env(env, n_steps=400, verbose=False)

    actions = buf.actions[: buf._size]
    # Simplex constraints
    np.testing.assert_allclose(actions.sum(axis=1), 1.0, atol=1e-5)
    assert (actions >= 0).all()
    # Uniform Dirichlet(1) -> each coord mean is 1/n; tolerate sampling noise.
    assert np.allclose(actions.mean(axis=0), 1 / 8, atol=0.05)


def test_mixture_sampling_proportions(buffer_factory):
    """Weighted-mixture sampling should reproduce the configured weights to
    within ~5% over 1000 episodes.
    """
    action_dim = 3
    # Each policy emits a different first-coordinate fingerprint.
    pol_a = _make_const_policy(0.7, action_dim)
    pol_b = _make_const_policy(0.1, action_dim)
    pol_c = _make_const_policy(0.4, action_dim)

    episode_length = 5
    n_episodes = 1000
    n_steps = episode_length * n_episodes

    env = _StubEnv(action_dim=action_dim, episode_length=episode_length)
    buf = buffer_factory(capacity=n_steps + 10, action_dim=action_dim)
    weights = [(pol_a, 0.5), (pol_b, 0.3), (pol_c, 0.2)]
    buf.load_from_env(
        env,
        n_steps=n_steps,
        policy_mixture=weights,
        verbose=False,
        mixture_seed=123,
    )

    actions = buf.actions[: buf._size]
    # The first coordinate uniquely identifies the policy that produced the
    # transition: 0.7 -> A, 0.1 -> B, 0.4 -> C.
    fingerprint = actions[:, 0]
    n_a = int(np.isclose(fingerprint, 0.7, atol=1e-3).sum())
    n_b = int(np.isclose(fingerprint, 0.1, atol=1e-3).sum())
    n_c = int(np.isclose(fingerprint, 0.4, atol=1e-3).sum())
    total = n_a + n_b + n_c
    assert total == n_steps, "every transition must be tagged by a known policy"

    # 1000-episode counts are binomial; 5%-of-target tolerance is loose
    # enough to keep the test deterministic under mixture_seed=123.
    assert abs(n_a / total - 0.5) < 0.05, n_a / total
    assert abs(n_b / total - 0.3) < 0.05, n_b / total
    assert abs(n_c / total - 0.2) < 0.05, n_c / total


def test_mixture_policy_constant_within_episode(buffer_factory):
    """Within one episode, every step should come from the same policy.
    Verified by checking the first-coordinate fingerprint is constant across
    every contiguous block of `episode_length` steps.
    """
    action_dim = 3
    pol_a = _make_const_policy(0.7, action_dim)
    pol_b = _make_const_policy(0.1, action_dim)

    episode_length = 4
    n_episodes = 50
    n_steps = episode_length * n_episodes

    env = _StubEnv(action_dim=action_dim, episode_length=episode_length)
    buf = buffer_factory(capacity=n_steps + 10, action_dim=action_dim)
    buf.load_from_env(
        env,
        n_steps=n_steps,
        policy_mixture=[(pol_a, 1.0), (pol_b, 1.0)],
        verbose=False,
        mixture_seed=7,
    )

    actions = buf.actions[: buf._size]
    fingerprint = actions[:, 0]
    # Each episode block must be a single fingerprint value.
    blocks = fingerprint.reshape(n_episodes, episode_length)
    block_first = blocks[:, :1]
    assert np.allclose(blocks, block_first, atol=1e-6), (
        "policy changed mid-episode in at least one block"
    )
    # Both fingerprints should appear at least once with mixture_seed=7.
    assert {round(float(v), 1) for v in block_first.flatten()} == {0.1, 0.7}


def test_mutual_exclusivity_raises(buffer_factory):
    env = _StubEnv()
    buf = buffer_factory()

    def _trivial(_obs):
        return np.array([1, 0, 0], dtype=np.float32)

    with pytest.raises(ValueError, match="not both"):
        buf.load_from_env(
            env, n_steps=5,
            policy=_trivial,
            policy_mixture=[(_trivial, 1.0)],
            verbose=False,
        )


def test_malformed_mixture_raises(buffer_factory):
    env = _StubEnv()
    buf = buffer_factory()

    def _trivial(_obs):
        return np.array([1, 0, 0], dtype=np.float32)

    with pytest.raises(ValueError, match="non-empty"):
        buf.load_from_env(env, n_steps=5, policy_mixture=[], verbose=False)
    with pytest.raises(ValueError, match="callable"):
        buf.load_from_env(
            env, n_steps=5,
            policy_mixture=[("not_callable", 1.0)],  # type: ignore[list-item]
            verbose=False,
        )
    with pytest.raises(ValueError, match="positive"):
        buf.load_from_env(
            env, n_steps=5, policy_mixture=[(_trivial, -1.0)], verbose=False
        )
