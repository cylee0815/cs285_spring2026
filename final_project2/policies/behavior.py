"""Behavior policies for offline dataset generation.

Each policy produces portfolio weights on the simplex (non-negative, sum to 1).
Used to generate diverse offline RL datasets for IQL training.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EqualWeightPolicy",
    "DirichletPolicy",
    "MomentumPolicy",
    "RiskParityPolicy",
]


class EqualWeightPolicy:
    """Deterministic equal-weight portfolio: w_i = 1 / n_assets."""

    def __init__(self, n_assets: int) -> None:
        self.n_assets = n_assets
        self._weights = np.ones(n_assets, dtype=np.float64) / n_assets

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self._weights.copy()


class DirichletPolicy:
    """Random portfolio sampled from a symmetric Dirichlet distribution."""

    def __init__(self, n_assets: int, alpha: float = 1.0, seed: int = 0) -> None:
        self.n_assets = n_assets
        self.alpha = alpha
        self._rng = np.random.RandomState(seed)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        w = self._rng.dirichlet(np.full(self.n_assets, self.alpha))
        return w


class MomentumPolicy:
    """Momentum-based portfolio: weights proportional to softmax of rolling mean returns.

    This policy needs access to recent returns history, not just the state vector.
    Use ``get_action_from_returns`` with the returns history.
    """

    def __init__(self, n_assets: int, lookback: int = 60) -> None:
        self.n_assets = n_assets
        self.lookback = lookback

    def get_action_from_returns(self, returns_history: np.ndarray) -> np.ndarray:
        """Compute momentum weights from a (T, n_assets) returns history.

        Uses the last ``lookback`` rows. If fewer rows available, uses all.
        Uses cumulative log return over the window as the score so the
        softmax has meaningful spread; raw daily means are ~1e-4 and
        collapse to a uniform distribution.
        """
        window = returns_history[-self.lookback:]
        scores = np.sum(window, axis=0)
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted)
        weights = exp_scores / exp_scores.sum()
        return weights


class RiskParityPolicy:
    """Risk parity portfolio: weights inversely proportional to volatility.

    Uses rolling standard deviation of returns over the lookback window.
    """

    def __init__(self, n_assets: int, lookback: int = 60) -> None:
        self.n_assets = n_assets
        self.lookback = lookback

    def get_action_from_returns(self, returns_history: np.ndarray) -> np.ndarray:
        """Compute risk parity weights from a (T, n_assets) returns history."""
        window = returns_history[-self.lookback:]
        vols = np.std(window, axis=0, ddof=0)
        # Inverse vol; clip to avoid division by zero
        inv_vol = 1.0 / np.maximum(vols, 1e-10)
        weights = inv_vol / inv_vol.sum()
        return weights
