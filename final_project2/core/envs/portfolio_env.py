"""Gym-style portfolio allocation environment for offline RL.

The agent observes market features at each timestep and outputs portfolio
weights on the simplex. Reward is the log of one plus the post-cost
portfolio return, matching the spec in CLAUDE.md.

MDP:
    state:   feature vector x_t (from FeatureBundle.features)
    action:  portfolio weights w_t in R^n_assets, projected to simplex
    reward:  log(1 + w_t . R_{t+1} - lambda * ||w_t - w_{t-1}||_1)
    done:    when the dataset is exhausted or episode_length reached
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

__all__ = ["PortfolioEnv"]


class PortfolioEnv:
    """Deterministic portfolio allocation environment.

    Parameters
    ----------
    features:
        ``(T, F)`` float32 array of market features.
    forward_returns:
        ``(T, N_assets)`` float32 array. ``forward_returns[t]`` is the
        asset return vector *after* the agent acts on ``features[t]``.
    transaction_cost_lambda:
        Coefficient for L1 turnover penalty. Default 0.0.
    episode_length:
        Maximum number of steps per episode. If ``None`` (default), the
        episode runs until the dataset is exhausted (``T - 1`` steps).
        Set to a finite value for buffer-loading (random-start episodes).
    """

    def __init__(
        self,
        features: np.ndarray,
        forward_returns: np.ndarray,
        transaction_cost_lambda: float = 0.0,
        episode_length: int | None = None,
    ) -> None:
        if features.shape[0] != forward_returns.shape[0]:
            raise ValueError(
                f"features and forward_returns must have same number of rows, "
                f"got {features.shape[0]} vs {forward_returns.shape[0]}"
            )
        if features.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 rows for a meaningful episode, "
                f"got {features.shape[0]}"
            )

        self._features = np.asarray(features, dtype=np.float32)
        self._forward_returns = np.asarray(forward_returns, dtype=np.float32)
        self._lambda = float(transaction_cost_lambda)
        self._n_assets = forward_returns.shape[1]
        self._n_steps = features.shape[0]

        # Episode length: default to full dataset sweep.
        self.episode_length: int = (
            episode_length if episode_length is not None else self._n_steps - 1
        )

        # Gymnasium-compatible spaces (required by offline-RL pipeline).
        obs_dim = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_assets,), dtype=np.float32,
        )

        # Mutable state — set by reset()
        self._start: int = 0
        self._t: int = 0
        self._done: bool = True
        self._portfolio_value: float = 1.0
        self.prev_weights: np.ndarray = np.ones(self._n_assets) / self._n_assets
        # RNG for random-start episodes.
        self._rng = np.random.default_rng()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Parameters
        ----------
        seed:
            Optional RNG seed for reproducibility.
        options:
            ``{"randomize": False}`` forces a chronological sweep from
            index 0 (for deterministic backtesting). Default behaviour
            (``randomize=True`` or omitted) samples a random start so
            the offline buffer gets diverse episodes.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        randomize = True if options is None else options.get("randomize", True)
        if randomize:
            max_start = max(0, self._n_steps - self.episode_length - 1)
            self._start = int(self._rng.integers(0, max(1, max_start + 1)))
        else:
            self._start = 0

        self._t = 0
        self._done = False
        self._portfolio_value = 1.0
        self.prev_weights = np.ones(self._n_assets, dtype=np.float64) / self._n_assets
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step.

        Parameters
        ----------
        action:
            Raw weight vector in R^n_assets. Will be projected onto the
            simplex before use.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        weights = _project_to_simplex(action, self._n_assets)

        # Index into the dataset arrays.
        idx = min(self._start + self._t, self._n_steps - 1)

        # Post-cost simple return, then logged for the reward signal.
        port_return = float(np.dot(weights, self._forward_returns[idx]))
        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        simple_return = port_return - self._lambda * turnover
        # Clamp to avoid log(0)/NaN; -1 is the mathematical floor (total loss).
        clamped = max(simple_return, -0.9999)
        reward = float(np.log1p(clamped))

        # Update state
        self.prev_weights = weights.copy()
        self._portfolio_value *= (1.0 + simple_return)
        self._t += 1

        # Termination conditions:
        #   terminated: dataset exhausted (reached the end of available data)
        #   truncated:  episode_length reached (but data still available)
        data_exhausted = (self._start + self._t) >= self._n_steps - 1
        ep_limit = self._t >= self.episode_length
        terminated = data_exhausted
        truncated = ep_limit and not data_exhausted

        if terminated or truncated:
            self._done = True

        obs = self._get_obs() if not self._done else np.zeros_like(self._features[0])

        info = {
            "portfolio_value": self._portfolio_value,
            # Pre-log post-cost return; both keys hold the same value.
            # ``portfolio_simple_return`` is the explicit name; ``portfolio_return``
            # is kept as a legacy alias used by older scripts.
            "portfolio_return": simple_return,
            "portfolio_simple_return": simple_return,
            # Log return == reward; kept in info for older eval code.
            "portfolio_log_return": reward,
            "turnover": turnover,
            "transaction_cost": self._lambda * turnover,
            "executed_weights": weights.copy(),
        }

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        idx = min(self._start + self._t, self._n_steps - 1)
        return self._features[idx].copy()


# ---------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------


def _project_to_simplex(v: np.ndarray, n: int) -> np.ndarray:
    """Project a vector onto the probability simplex.

    1. Clip negatives to zero.
    2. If all zeros, return equal weight.
    3. Otherwise normalize to sum to 1.
    """
    w = np.maximum(np.asarray(v, dtype=np.float64).ravel()[:n], 0.0)
    total = w.sum()
    if total < 1e-12:
        return np.ones(n, dtype=np.float64) / n
    return w / total
