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
        accept_portfolio_weights: bool = False,
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
        # When True, treat the incoming action as already-normalized portfolio
        # weights on the simplex (e.g. DirichletActor.mean or an MLP softmax head)
        # and skip the softmax in step(). Fixes the double-softmax bug where
        # softmax(weights) collapses toward uniform.
        self.accept_portfolio_weights = accept_portfolio_weights

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
        # `randomize` controls whether we sample a random episode start or
        # sweep the dataset chronologically from index 0. Default is True so
        # existing (offline-buffer) callers get random-start episodes.
        randomize = True if options is None else options.get("randomize", True)
        if randomize:
            # Sample start index such that episode fits; bound safely so we
            # never pass a negative limit to numpy.
            max_start = self.T - self.episode_length - 1
            safe_high = max(1, max_start)
            self._start = int(self.np_random.integers(0, safe_high))
        else:
            # Chronological backtest: start at the first day of the split.
            self._start = 0
        self._t = 0
        # Equal-weight initialization
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._portfolio_value = 1.0
        self._sharpe_A = 0.0
        self._sharpe_B = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.array(action, dtype=np.float32)

        if self.accept_portfolio_weights:
            # Agent already outputs weights on the simplex — use them directly.
            raw = action[:self.n_assets]
            if np.any(np.isnan(raw)) or np.any(np.isinf(raw)) or np.any(raw < 0):
                new_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
            else:
                s = float(raw.sum())
                if s <= 1e-8:
                    new_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
                else:
                    new_weights = (raw / s).astype(np.float32)
        else:
            # Agent outputs logits — convert via softmax.
            action = np.clip(action, -20.0, 20.0)
            if self.include_cash:
                new_weights_with_cash = softmax(action)
                new_weights = new_weights_with_cash[:self.n_assets].astype(np.float32)
            else:
                new_weights = softmax(action[:self.n_assets]).astype(np.float32)
            if np.any(np.isnan(new_weights)) or np.any(np.isinf(new_weights)):
                new_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        # Transaction cost: L1 distance between old and new weights
        turnover = float(np.sum(np.abs(new_weights - self._weights)))
        tc = self.transaction_cost * turnover

        # Get returns for this step — clamp index to valid range
        idx = min(self._start + self._t, self.T - 1)
        log_r = self.price_returns[idx]  # (n_assets,) log returns
        log_r = np.where(np.isnan(log_r) | np.isinf(log_r), 0.0, log_r)

        # Exact portfolio simple return from per-asset log returns, minus
        # proportional transaction cost. Mixing log and simple returns in the
        # old code produced O(sigma^2/2) drift and inconsistent PV tracking.
        simple_r = np.expm1(log_r)                            # exp(log_r) - 1
        port_simple = float(np.dot(new_weights, simple_r)) - tc
        # log return of the portfolio step (safe for port_simple <= -1)
        port_log = float(np.log1p(max(port_simple, -0.999999)))

        if self.reward_type == "log_return":
            reward = port_log
        elif self.reward_type == "diff_sharpe":
            reward = self._differential_sharpe(port_simple)
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        # Update state — compound PV with the consistent simple return
        self._weights = new_weights
        self._portfolio_value *= (1.0 + port_simple)
        self._t += 1

        terminated = False
        truncated = self._t >= self.episode_length

        info = {
            "portfolio_value": self._portfolio_value,
            "portfolio_return": port_simple,
            "portfolio_log_return": port_log,
            "turnover": turnover,
            "transaction_cost": tc,
            # The actual weights used to compute returns this step. Offline
            # buffers should store THIS (not the raw action) so downstream
            # algorithms train Q/actor on the true behavioral policy.
            "executed_weights": new_weights.copy(),
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
