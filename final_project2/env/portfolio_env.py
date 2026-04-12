"""Gym-style portfolio allocation environment for offline RL.

The agent observes market features at each timestep and outputs portfolio
weights on the simplex. Reward is the dot product of weights and forward
returns, minus an L1 turnover penalty.

MDP:
    state:   feature vector x_t (from FeatureBundle.features)
    action:  portfolio weights w_t in R^n_assets, projected to simplex
    reward:  w_t . R_{t+1} - lambda * ||w_t - w_{t-1}||_1
    done:    when the dataset is exhausted
"""

from __future__ import annotations

import numpy as np

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
    """

    def __init__(
        self,
        features: np.ndarray,
        forward_returns: np.ndarray,
        transaction_cost_lambda: float = 0.0,
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

        # Mutable state — set by reset()
        self._t: int = 0
        self._done: bool = True
        self.prev_weights: np.ndarray = np.ones(self._n_assets) / self._n_assets

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset to the beginning of the dataset.

        Parameters
        ----------
        seed:
            Accepted for API compatibility. The environment is fully
            deterministic (fixed dataset), so the seed has no effect on
            the trajectory. It is stored but not used.
        """
        self._t = 0
        self._done = False
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

        # Reward: portfolio return minus turnover cost
        port_return = float(np.dot(weights, self._forward_returns[self._t]))
        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        reward = port_return - self._lambda * turnover

        # Update state
        self.prev_weights = weights.copy()
        self._t += 1

        # Check done: we can act on rows 0..T-2; after stepping from T-2
        # we've consumed T-1 transitions total, so the episode ends.
        terminated = self._t >= self._n_steps - 1
        if terminated:
            self._done = True

        obs = self._get_obs() if not terminated else np.zeros_like(self._features[0])

        return obs, reward, terminated, False, {}

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return self._features[self._t].copy()


# ---------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------


def _project_to_simplex(v: np.ndarray, n: int) -> np.ndarray:
    """Project a vector onto the probability simplex.

    1. Clip negatives to zero.
    2. If all zeros, return equal weight.
    3. Otherwise normalize to sum to 1.
    """
    w = np.maximum(v.astype(np.float64), 0.0)
    total = w.sum()
    if total < 1e-12:
        return np.ones(n, dtype=np.float64) / n
    return w / total
