"""
PortfolioEnv: Gymnasium environment for portfolio optimization.

State: concatenation of:
  [current_weights (n_assets),
   flattened per-asset features (n_assets × n_features),    ← price/TA
   global macro features (n_macro),                          ← FRED: rates, CPI, etc.
   sentiment features (n_sentiment)]                         ← SF Fed DNSI / Alpaca

Per-asset features: [log_return, rolling_mean_return, rolling_std_return, rsi, macd, bb_pct]
Macro features (optional, N_MACRO_FEATURES=8): dgs10, dff, yield_spread, cpi_yoy, unrate,
  gdp_growth, umcsent_norm, nfci
Sentiment features (optional): SF Fed DNSI scalar (+ Alpaca embeddings if precomputed)

Action: raw logits of shape (n_assets,), converted to portfolio weights via softmax.
  (supports a cash position by having n_assets+1 action dim with last = cash)

Reward: either log portfolio return or differential Sharpe ratio.

Episode: fixed-length window sampled from historical data (train split).
  At reset, randomly sample a start index from the train period.
  At each step, advance one trading day.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from scipy.special import softmax


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        price_returns: np.ndarray,              # (T, n_assets) log returns
        features: np.ndarray,                    # (T, n_assets, n_features) per-asset TA
        episode_length: int = 63,
        transaction_cost: float = 0.001,
        reward_type: str = "log_return",         # "log_return" or "diff_sharpe"
        include_cash: bool = False,
        sharpe_eta: float = 0.01,
        macro_features: Optional[np.ndarray] = None,     # (T, n_macro) global macro
        sentiment_features: Optional[np.ndarray] = None, # (T, n_sentiment) news sentiment
    ):
        super().__init__()
        self.price_returns = price_returns.astype(np.float32)  # (T, n_assets)
        self.features = features.astype(np.float32)             # (T, n_assets, n_features)
        self.macro_features = (
            macro_features.astype(np.float32) if macro_features is not None else None
        )
        self.sentiment_features = (
            sentiment_features.astype(np.float32) if sentiment_features is not None else None
        )

        self.T, self.n_assets = price_returns.shape
        self.n_features = features.shape[2]
        self.n_macro = macro_features.shape[1] if macro_features is not None else 0
        self.n_sentiment = sentiment_features.shape[1] if sentiment_features is not None else 0
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type
        self.include_cash = include_cash
        self.sharpe_eta = sharpe_eta

        self.n_actions = self.n_assets + (1 if include_cash else 0)

        # Observation: [weights(n_assets), per-asset features(n×f), macro(m), sentiment(s)]
        obs_dim = (
            self.n_assets
            + self.n_assets * self.n_features
            + self.n_macro
            + self.n_sentiment
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: logits for softmax (unnormalized weights)
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_actions,), dtype=np.float32
        )

        self._t = 0
        self._start = 0
        self._weights = None
        # Differential Sharpe state
        self._sharpe_A = 0.0
        self._sharpe_B = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Sample start index such that episode fits
        max_start = self.T - self.episode_length - 1
        self._start = int(self.np_random.integers(0, max(1, max_start)))
        self._t = 0
        # Equal-weight initialization
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._portfolio_value = 1.0
        self._sharpe_A = 0.0
        self._sharpe_B = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Convert action logits to weights via softmax
        action = np.array(action, dtype=np.float32)
        # Clip to avoid overflow in softmax
        action = np.clip(action, -20.0, 20.0)

        if self.include_cash:
            new_weights_with_cash = softmax(action)
            new_weights = new_weights_with_cash[:self.n_assets].astype(np.float32)
        else:
            new_weights = softmax(action[:self.n_assets]).astype(np.float32)

        # Handle NaN weights (fall back to equal weights)
        if np.any(np.isnan(new_weights)) or np.any(np.isinf(new_weights)):
            new_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        # Transaction cost: L1 distance between old and new weights
        turnover = float(np.sum(np.abs(new_weights - self._weights)))
        tc = self.transaction_cost * turnover

        # Get returns for this step — clamp index to valid range
        idx = min(self._start + self._t, self.T - 1)
        returns = self.price_returns[idx]  # (n_assets,)

        # Replace any NaN returns with 0
        returns = np.where(np.isnan(returns), 0.0, returns)

        # Portfolio return
        port_return = float(np.dot(new_weights, returns)) - tc

        # Compute reward
        if self.reward_type == "log_return":
            reward = float(port_return)
        elif self.reward_type == "diff_sharpe":
            reward = self._differential_sharpe(port_return)
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        # Update state
        self._weights = new_weights
        self._portfolio_value *= (1.0 + port_return)
        self._t += 1

        terminated = False
        truncated = self._t >= self.episode_length

        info = {
            "portfolio_value": self._portfolio_value,
            "portfolio_return": port_return,
            "turnover": turnover,
            "transaction_cost": tc,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        idx = min(self._start + self._t, self.T - 1)
        feat = self.features[idx]  # (n_assets, n_features)
        feat = np.where(np.isnan(feat), 0.0, feat)

        parts = [self._weights, feat.flatten()]

        if self.macro_features is not None:
            macro = self.macro_features[idx]
            macro = np.where(np.isnan(macro), 0.0, macro)
            parts.append(macro)

        if self.sentiment_features is not None:
            sent = self.sentiment_features[idx]
            sent = np.where(np.isnan(sent), 0.0, sent)
            parts.append(sent)

        return np.concatenate(parts, axis=0).astype(np.float32)

    def _differential_sharpe(self, r: float) -> float:
        """Moody & Saffell (2001) differential Sharpe ratio."""
        eta = self.sharpe_eta
        delta_A = r - self._sharpe_A
        delta_B = r ** 2 - self._sharpe_B
        denom = (self._sharpe_B - self._sharpe_A ** 2) ** 1.5
        if denom < 1e-8:
            dsr = 0.0
        else:
            dsr = (self._sharpe_B * delta_A - 0.5 * self._sharpe_A * delta_B) / denom
        # Update EMA
        self._sharpe_A += eta * delta_A
        self._sharpe_B += eta * delta_B
        return float(np.clip(dsr, -1.0, 1.0))

    def render(self):
        pass
