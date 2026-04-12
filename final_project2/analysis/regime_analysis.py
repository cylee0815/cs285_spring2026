"""Regime-split analysis of backtest results.

Splits a test-period equity curve and returns series into market regimes
and computes per-regime performance metrics.

Usage
-----
    python analysis/regime_analysis.py \
        --returns results/default/portfolio_returns.npy \
        --weights results/default/weights.npy \
        --start_date 2008-01-01

    # Or from a backtest results directory with metrics.csv:
    python analysis/regime_analysis.py --results_dir results/default --start_date 2022-01-01
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from evaluation.metrics import (
    compute_annual_return,
    compute_annual_volatility,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
)

# Default regime definitions (approximate trading-day indices).
# These assume daily data starting from 2008-01-01.
# Adjust start_date to shift regime boundaries.
REGIMES = {
    "financial_crisis": ("2008-01-01", "2009-12-31"),
    "normal_market": ("2010-01-01", "2019-12-31"),
    "covid_crash": ("2020-01-01", "2020-12-31"),
    "modern_regime": ("2021-01-01", "2026-12-31"),
}


def _date_to_index(start_date: str, target_date: str) -> int:
    """Convert a date string to an approximate trading-day index.

    Assumes ~252 trading days per year relative to start_date.
    """
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d")
    target = datetime.strptime(target_date, "%Y-%m-%d")
    calendar_days = (target - start).days
    # Approximate: 252 trading days / 365 calendar days
    return max(0, int(calendar_days * 252 / 365))


def compute_regime_metrics(
    portfolio_returns: np.ndarray,
    weights: np.ndarray,
    start_date: str = "2008-01-01",
    regimes: dict[str, tuple[str, str]] | None = None,
) -> list[dict[str, float | str]]:
    """Compute per-regime metrics.

    Parameters
    ----------
    portfolio_returns:
        1-D array of daily portfolio returns.
    weights:
        2-D array of shape (T, n_assets).
    start_date:
        Calendar date corresponding to index 0.
    regimes:
        Dict mapping regime name to (start_date, end_date) strings.
        Defaults to standard financial regimes.

    Returns
    -------
    List of dicts, one per regime with metrics.
    """
    if regimes is None:
        regimes = REGIMES

    T = len(portfolio_returns)
    results = []

    for regime_name, (r_start, r_end) in regimes.items():
        idx_start = _date_to_index(start_date, r_start)
        idx_end = _date_to_index(start_date, r_end)

        # Clip to available data
        idx_start = min(idx_start, T)
        idx_end = min(idx_end, T)

        if idx_start >= idx_end or idx_start >= T:
            continue

        regime_returns = portfolio_returns[idx_start:idx_end]
        regime_weights = weights[idx_start:idx_end]
        regime_equity = np.cumprod(1 + regime_returns)

        row = {
            "regime": regime_name,
            "start_idx": idx_start,
            "end_idx": idx_end,
            "n_days": len(regime_returns),
            "annual_return": compute_annual_return(regime_returns),
            "annual_volatility": compute_annual_volatility(regime_returns),
            "sharpe_ratio": compute_sharpe(regime_returns),
            "max_drawdown": compute_max_drawdown(regime_equity),
            "turnover": compute_turnover(regime_weights),
            "cumulative_return": float(np.prod(1 + regime_returns) - 1),
        }
        results.append(row)

    return results


def write_regime_csv(rows: list[dict], output_path: Path) -> None:
    """Write regime metrics to CSV."""
    if not rows:
        print("No regime data to write.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Regime metrics saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-split analysis.")
    parser.add_argument("--returns", type=str, default=None,
                        help="Path to portfolio_returns .npy file.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights .npy file.")
    parser.add_argument("--start_date", type=str, default="2008-01-01",
                        help="Calendar date corresponding to index 0.")
    parser.add_argument("--output", type=str, default="results/regime_metrics.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    if args.returns is None:
        print("Error: --returns is required (path to portfolio_returns .npy file)")
        sys.exit(1)

    portfolio_returns = np.load(args.returns)
    if args.weights is not None:
        weights = np.load(args.weights)
    else:
        # Dummy weights for metric computation
        weights = np.ones((len(portfolio_returns), 8)) / 8

    rows = compute_regime_metrics(portfolio_returns, weights, start_date=args.start_date)
    write_regime_csv(rows, Path(args.output))

    # Print summary
    print(f"\n{'Regime':<20} {'Days':>6} {'AnnRet':>10} {'Sharpe':>10} {'MDD':>10}")
    print("-" * 60)
    for r in rows:
        print(f"{r['regime']:<20} {r['n_days']:>6} {r['annual_return']:>+10.4f} "
              f"{r['sharpe_ratio']:>10.4f} {r['max_drawdown']:>10.4f}")


if __name__ == "__main__":
    main()
