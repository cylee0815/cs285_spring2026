"""Core financial performance metrics.

All functions operate on NumPy arrays and return scalar values.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_sharpe",
    "compute_max_drawdown",
    "compute_turnover",
    "compute_cumulative_return",
    "compute_annual_return",
    "compute_annual_volatility",
    "compute_rolling_sharpe",
]


def compute_sharpe(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Annualized Sharpe ratio (assuming zero risk-free rate).

    Parameters
    ----------
    returns:
        1-D array of periodic (e.g. daily) portfolio returns.
    annualization:
        Trading days per year. Default 252.

    Returns
    -------
    Scalar Sharpe ratio.  Returns 0.0 if volatility is zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size == 0:
        return 0.0
    std = float(np.std(returns, ddof=0))
    if std < 1e-15:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(annualization))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown.

    Parameters
    ----------
    equity_curve:
        1-D array of cumulative portfolio values (e.g. starting at 1.0).

    Returns
    -------
    Non-negative scalar. 0.0 means the equity curve never declined.
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / np.where(
        running_max > 0, running_max, 1.0
    )
    return float(np.max(drawdowns))


def compute_turnover(weights: np.ndarray) -> float:
    """Mean L1 turnover between consecutive weight vectors.

    Parameters
    ----------
    weights:
        2-D array of shape ``(T, n_assets)`` — portfolio weights at each step.

    Returns
    -------
    Mean L1 distance between consecutive weight rows.
    Returns 0.0 if there are fewer than 2 rows.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] < 2:
        return 0.0
    diffs = np.abs(weights[1:] - weights[:-1]).sum(axis=1)
    return float(np.mean(diffs))


def compute_cumulative_return(returns: np.ndarray) -> float:
    """Total cumulative return: prod(1 + r_t) - 1.

    Parameters
    ----------
    returns:
        1-D array of periodic returns.
    """
    returns = np.asarray(returns, dtype=np.float64)
    return float(np.prod(1 + returns) - 1)


def compute_annual_return(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Annualized mean return.

    Parameters
    ----------
    returns:
        1-D array of daily returns.
    annualization:
        Trading days per year.
    """
    returns = np.asarray(returns, dtype=np.float64)
    return float(np.mean(returns) * annualization)


def compute_annual_volatility(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Annualized volatility.

    Parameters
    ----------
    returns:
        1-D array of daily returns.
    annualization:
        Trading days per year.
    """
    returns = np.asarray(returns, dtype=np.float64)
    return float(np.std(returns, ddof=0) * np.sqrt(annualization))


def compute_rolling_sharpe(
    returns: np.ndarray,
    window: int = 63,
    annualization: float = 252.0,
) -> np.ndarray:
    """Rolling Sharpe ratio.

    Parameters
    ----------
    returns:
        1-D array of daily returns.
    window:
        Rolling window length (default 63 ~ one quarter).
    annualization:
        Trading days per year.

    Returns
    -------
    1-D array of length ``len(returns)``. Early values (before ``window``
    observations are available) are NaN.
    """
    returns = np.asarray(returns, dtype=np.float64)
    T = len(returns)
    result = np.full(T, np.nan)
    for t in range(window, T + 1):
        chunk = returns[t - window : t]
        std = float(np.std(chunk, ddof=0))
        if std < 1e-15:
            result[t - 1] = 0.0
        else:
            result[t - 1] = float(np.mean(chunk) / std * np.sqrt(annualization))
    return result


def compute_all_metrics(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Compute a standard suite of metrics and return as a flat dict."""
    return {
        "annual_return": compute_annual_return(returns),
        "annual_volatility": compute_annual_volatility(returns),
        "sharpe_ratio": compute_sharpe(returns),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "cumulative_return": compute_cumulative_return(returns),
        "turnover": compute_turnover(weights),
    }
