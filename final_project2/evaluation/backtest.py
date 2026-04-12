"""Walk-forward backtest runner for trained IQL agents.

Runs the policy through a PortfolioEnv (or raw state/return arrays) and
records portfolio returns, weights, and equity curve.
"""

from __future__ import annotations

import numpy as np
import torch

from algorithms.iql import IQL
from env.portfolio_env import PortfolioEnv
from evaluation.metrics import compute_all_metrics

__all__ = ["run_backtest", "run_backtest_from_arrays"]


def run_backtest(
    agent: IQL,
    env: PortfolioEnv,
    device: str = "cpu",
) -> dict[str, np.ndarray | dict[str, float]]:
    """Roll out a trained agent through a PortfolioEnv episode.

    Parameters
    ----------
    agent:
        Trained IQL agent.
    env:
        PortfolioEnv instance (typically built from the test split).
    device:
        Torch device for inference.

    Returns
    -------
    Dict with keys: ``portfolio_returns``, ``weights``, ``equity_curve``,
    ``metrics``.
    """
    obs, _ = env.reset()
    n_assets = env._n_assets

    weights_list: list[np.ndarray] = []
    returns_list: list[float] = []

    terminated = False
    while not terminated:
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = agent.get_action(state_tensor)
        w = action.cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(w)

        weights_list.append(w.copy())
        returns_list.append(reward)

    weights = np.array(weights_list, dtype=np.float64)
    portfolio_returns = np.array(returns_list, dtype=np.float64)
    equity_curve = np.cumprod(1 + portfolio_returns)

    metrics = compute_all_metrics(portfolio_returns, equity_curve, weights)

    return {
        "portfolio_returns": portfolio_returns,
        "weights": weights,
        "equity_curve": equity_curve,
        "metrics": metrics,
    }


def run_backtest_from_arrays(
    agent: IQL,
    states: np.ndarray,
    forward_returns: np.ndarray,
    transaction_cost_lambda: float = 0.001,
    device: str = "cpu",
) -> dict[str, np.ndarray | dict[str, float]]:
    """Run a backtest directly from state and return arrays.

    This is a convenience wrapper that constructs a PortfolioEnv and calls
    :func:`run_backtest`.

    Parameters
    ----------
    agent:
        Trained IQL agent.
    states:
        ``(T, state_dim)`` array of observations.
    forward_returns:
        ``(T, n_assets)`` array of forward asset returns.
    transaction_cost_lambda:
        L1 turnover penalty coefficient.
    device:
        Torch device for inference.
    """
    env = PortfolioEnv(
        features=states,
        forward_returns=forward_returns,
        transaction_cost_lambda=transaction_cost_lambda,
    )
    return run_backtest(agent, env, device=device)
