"""Backtest a trained IQL agent on a contiguous date-indexed slice.

Separates the "which transitions belong to this period" question (handled
via :mod:`data.splits`) from the roll-out logic, so the same helper is
used by the in-training validation loop and by the final test evaluation.
"""

from __future__ import annotations

import numpy as np

from algorithms.iql import IQL
from env.portfolio_env import PortfolioEnv
from evaluation.backtest import run_backtest

__all__ = ["backtest_on_indices"]


def backtest_on_indices(
    agent: IQL,
    states: np.ndarray,
    forward_returns: np.ndarray,
    indices: np.ndarray,
    transaction_cost_lambda: float,
    device: str,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Run a backtest on a contiguous sub-window of the dataset.

    Parameters
    ----------
    agent:
        Trained IQL agent.
    states:
        Full ``(T, state_dim)`` state array from the dataset.
    forward_returns:
        Full ``(T, n_assets)`` forward-returns array aligned with
        ``states``.
    indices:
        1-D array of integer indices (into the dataset) identifying the
        window to evaluate on. Must be sorted ascending and contiguous
        *within a calendar period*; gaps are allowed between periods but
        this function treats the window as a single episode.
    transaction_cost_lambda:
        L1 turnover penalty used for reward/backtest consistency.
    device:
        Torch device for inference.

    Returns
    -------
    Dict with keys ``portfolio_returns``, ``weights``, ``equity_curve``,
    ``metrics`` — same schema as :func:`evaluation.backtest.run_backtest`.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size < 2:
        raise ValueError(
            f"Need at least 2 indices for a backtest window, got {idx.size}."
        )

    states_slice = np.ascontiguousarray(states[idx], dtype=np.float32)
    returns_slice = np.ascontiguousarray(forward_returns[idx], dtype=np.float32)

    env = PortfolioEnv(
        features=states_slice,
        forward_returns=returns_slice,
        transaction_cost_lambda=transaction_cost_lambda,
    )
    return run_backtest(agent, env, device=device)
