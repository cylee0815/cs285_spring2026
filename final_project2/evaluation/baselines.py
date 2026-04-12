"""Classical portfolio baseline strategies for comparison.

Each function takes the environment's forward-returns matrix and returns
the same ``(returns, weights, equity_curve)`` structure as the IQL
backtest so they can be compared on equal footing.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "equal_weight_backtest",
    "momentum_backtest",
    "risk_parity_backtest",
    "buy_and_hold_backtest",
]


def equal_weight_backtest(
    forward_returns: np.ndarray,
    transaction_cost_lambda: float = 0.0,
) -> dict[str, np.ndarray]:
    """Equal-weight (1/N) portfolio held every period.

    Parameters
    ----------
    forward_returns:
        ``(T, n_assets)`` array of per-period asset returns.
    transaction_cost_lambda:
        L1 turnover cost coefficient.
    """
    T, n_assets = forward_returns.shape
    w = np.ones(n_assets, dtype=np.float64) / n_assets
    weights = np.tile(w, (T, 1))
    port_returns = forward_returns @ w  # no turnover after first step
    # First step has turnover from assumed equal-weight starting position → 0
    equity_curve = np.cumprod(1 + port_returns)
    return {
        "portfolio_returns": port_returns,
        "weights": weights,
        "equity_curve": equity_curve,
    }


def momentum_backtest(
    forward_returns: np.ndarray,
    lookback: int = 60,
    transaction_cost_lambda: float = 0.0,
) -> dict[str, np.ndarray]:
    """Momentum portfolio: weights proportional to softmax of trailing mean returns.

    Parameters
    ----------
    forward_returns:
        ``(T, n_assets)`` array.
    lookback:
        Number of past periods for the momentum signal.
    transaction_cost_lambda:
        L1 turnover cost coefficient.
    """
    T, n_assets = forward_returns.shape
    weights = np.zeros((T, n_assets), dtype=np.float64)
    port_returns = np.zeros(T, dtype=np.float64)
    prev_w = np.ones(n_assets, dtype=np.float64) / n_assets

    for t in range(T):
        if t < lookback:
            w = np.ones(n_assets, dtype=np.float64) / n_assets
        else:
            scores = np.mean(forward_returns[t - lookback : t], axis=0)
            scores_shifted = scores - np.max(scores)
            exp_s = np.exp(scores_shifted)
            w = exp_s / exp_s.sum()

        turnover = np.abs(w - prev_w).sum()
        port_returns[t] = float(np.dot(w, forward_returns[t])) - transaction_cost_lambda * turnover
        weights[t] = w
        prev_w = w

    equity_curve = np.cumprod(1 + port_returns)
    return {
        "portfolio_returns": port_returns,
        "weights": weights,
        "equity_curve": equity_curve,
    }


def risk_parity_backtest(
    forward_returns: np.ndarray,
    lookback: int = 60,
    transaction_cost_lambda: float = 0.0,
) -> dict[str, np.ndarray]:
    """Risk-parity portfolio: weights inversely proportional to trailing volatility.

    Parameters
    ----------
    forward_returns:
        ``(T, n_assets)`` array.
    lookback:
        Window for volatility estimation.
    transaction_cost_lambda:
        L1 turnover cost coefficient.
    """
    T, n_assets = forward_returns.shape
    weights = np.zeros((T, n_assets), dtype=np.float64)
    port_returns = np.zeros(T, dtype=np.float64)
    prev_w = np.ones(n_assets, dtype=np.float64) / n_assets

    for t in range(T):
        if t < lookback:
            w = np.ones(n_assets, dtype=np.float64) / n_assets
        else:
            vols = np.std(forward_returns[t - lookback : t], axis=0, ddof=0)
            inv_vol = 1.0 / np.maximum(vols, 1e-10)
            w = inv_vol / inv_vol.sum()

        turnover = np.abs(w - prev_w).sum()
        port_returns[t] = float(np.dot(w, forward_returns[t])) - transaction_cost_lambda * turnover
        weights[t] = w
        prev_w = w

    equity_curve = np.cumprod(1 + port_returns)
    return {
        "portfolio_returns": port_returns,
        "weights": weights,
        "equity_curve": equity_curve,
    }


def buy_and_hold_backtest(
    forward_returns: np.ndarray,
) -> dict[str, np.ndarray]:
    """Buy-and-hold equal-weight portfolio (no rebalancing).

    Weights drift with asset returns after the initial equal-weight allocation.

    Parameters
    ----------
    forward_returns:
        ``(T, n_assets)`` array.
    """
    T, n_assets = forward_returns.shape
    weights = np.zeros((T, n_assets), dtype=np.float64)
    port_returns = np.zeros(T, dtype=np.float64)

    w = np.ones(n_assets, dtype=np.float64) / n_assets

    for t in range(T):
        weights[t] = w
        port_returns[t] = float(np.dot(w, forward_returns[t]))
        # Drift weights according to individual asset returns
        w = w * (1 + forward_returns[t])
        total = w.sum()
        if total > 1e-12:
            w = w / total
        else:
            w = np.ones(n_assets, dtype=np.float64) / n_assets

    equity_curve = np.cumprod(1 + port_returns)
    return {
        "portfolio_returns": port_returns,
        "weights": weights,
        "equity_curve": equity_curve,
    }
