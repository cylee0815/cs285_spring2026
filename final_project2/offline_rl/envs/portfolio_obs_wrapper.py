"""
PortfolioObsWrapper: augments observations with previous portfolio weights.

Without this wrapper the environment is partially observable — the agent
cannot see the current allocation, yet transaction costs depend on it.
Prepending prev_weights to the feature vector restores the Markov property.

    obs = [prev_weights (action_dim), market_features (feature_dim)]

The wrapper does NOT modify the base environment. It reads
``info["executed_weights"]`` from env.step() to track allocations.

Note: PortfolioEnv is *not* a gymnasium.Env subclass, so this wrapper
delegates manually rather than using gymnasium.Wrapper.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces


class PortfolioObsWrapper:
    """Prepend previous portfolio weights to the observation vector.

    Parameters
    ----------
    env:
        A PortfolioEnv (or any env whose ``step()`` returns
        ``info["executed_weights"]`` as an ndarray of portfolio weights).
    """

    def __init__(self, env) -> None:
        self.env = env

        self._action_dim = env.action_space.shape[0]
        base_obs_dim = env.observation_space.shape[0]
        aug_dim = self._action_dim + base_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(aug_dim,), dtype=np.float32,
        )
        # Action space is unchanged — pass through from inner env.
        self.action_space = env.action_space

        self._prev_weights = np.ones(self._action_dim, dtype=np.float32) / self._action_dim

    def __getattr__(self, name):
        """Delegate attribute access to the inner env for episode_length etc."""
        return getattr(self.env, name)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([self._prev_weights, obs]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_weights = np.ones(self._action_dim, dtype=np.float32) / self._action_dim
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Update prev_weights from the env's actually-executed weights
        executed = info.get("executed_weights")
        if executed is not None:
            self._prev_weights = np.asarray(executed, dtype=np.float32)
        return self._augment(obs), reward, terminated, truncated, info
