"""Regime-split analysis of backtest results.

Splits a backtest equity curve / returns series into calendar regimes
(financial crisis, normal market, COVID crash, modern regime, …) and
computes per-regime performance metrics.

Usage
-----
    # Direct arrays (expects portfolio_returns.npy, weights.npy, optional dates.npy)
    python analysis/regime_analysis.py \
        --returns results/default/test/portfolio_returns.npy \
        --weights results/default/test/weights.npy \
        --dates   results/default/test/dates.npy

    # Or point at a directory containing the canonical artifact set:
    python analysis/regime_analysis.py --results_dir results/default/test

The ``--results_dir`` form is preferred because the upstream logger now
writes ``weights.npy``, ``portfolio_returns.npy``, and ``dates.npy``
side-by-side in both ``run_dir/validation/`` and ``run_dir/test/``.
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

# Default regime definitions.
REGIMES = {
    "financial_crisis": ("2008-01-01", "2009-12-31"),
    "normal_market": ("2010-01-01", "2019-12-31"),
    "covid_crash": ("2020-01-01", "2020-12-31"),
    "modern_regime": ("2021-01-01", "2026-12-31"),
}


# ---------------------------------------------------------------------------
# Index mapping helpers
# ---------------------------------------------------------------------------


def _calendar_to_index(start_date: str, target_date: str) -> int:
    """Approximate calendar→index conversion (legacy fallback).

    Assumes ~252 trading days per 365 calendar days. Used only when no
    ``dates`` array is available; with real dates prefer :func:`_real_dates_to_index`.
    """
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d")
    target = datetime.strptime(target_date, "%Y-%m-%d")
    calendar_days = (target - start).days
    return max(0, int(calendar_days * 252 / 365))


def _real_dates_to_index(dates_ns: np.ndarray, target_date: str) -> int:
    """Find the first index whose date is >= target_date.

    ``dates_ns`` is assumed monotone non-decreasing (enforced by the
    training script's split assertions). Binary-searches the first
    element that crosses the regime boundary.
    """
    import pandas as pd

    target_ns = np.int64(pd.Timestamp(target_date).value)
    return int(np.searchsorted(dates_ns, target_ns, side="left"))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_regime_metrics(
    portfolio_returns: np.ndarray,
    weights: np.ndarray,
    dates: np.ndarray | None = None,
    start_date: str = "2008-01-01",
    regimes: dict[str, tuple[str, str]] | None = None,
) -> list[dict[str, float | str]]:
    """Compute per-regime metrics.

    Parameters
    ----------
    portfolio_returns:
        1-D array of per-period portfolio returns.
    weights:
        2-D array of shape ``(T, n_assets)`` — aligned with ``portfolio_returns``.
    dates:
        Optional ``(T,)`` int64 array of nanosecond-resolution timestamps
        aligned with the other two arrays. When provided, regime boundaries
        are mapped to array indices using the true dates (recommended).
        When ``None``, falls back to the approximate calendar->index mapping.
    start_date:
        Only used when ``dates`` is ``None`` — the calendar date of index 0
        in the returns array.
    regimes:
        Dict mapping regime name to ``(start_date, end_date)`` strings.
    """
    if regimes is None:
        regimes = REGIMES

    T = len(portfolio_returns)
    assert weights.shape[0] == T, (
        f"weights.shape[0]={weights.shape[0]} does not match "
        f"len(portfolio_returns)={T}"
    )
    if dates is not None:
        assert dates.shape[0] == T, (
            f"dates length {dates.shape[0]} does not match returns length {T}"
        )

    results: list[dict[str, float | str]] = []

    for regime_name, (r_start, r_end) in regimes.items():
        if dates is not None:
            idx_start = _real_dates_to_index(dates, r_start)
            # end bound is inclusive of r_end — use strictly-greater marker.
            idx_end = _real_dates_to_index(dates, r_end)
            # Include the last trading day of r_end: if dates[idx_end] matches
            # r_end we want it inside the window.
            if idx_end < T:
                import pandas as pd
                if int(dates[idx_end]) <= int(pd.Timestamp(r_end).value):
                    idx_end += 1
        else:
            idx_start = _calendar_to_index(start_date, r_start)
            idx_end = _calendar_to_index(start_date, r_end)

        idx_start = min(max(idx_start, 0), T)
        idx_end = min(max(idx_end, 0), T)
        if idx_start >= idx_end or idx_start >= T:
            continue

        regime_returns = portfolio_returns[idx_start:idx_end]
        regime_weights = weights[idx_start:idx_end]
        regime_equity = np.cumprod(1 + regime_returns)

        results.append({
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
        })

    return results


def write_regime_csv(rows: list[dict], output_path: Path) -> None:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_inputs(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load portfolio_returns, weights, (optional) dates from CLI args.

    Supports two mutually exclusive modes:

    * ``--results_dir`` — read the canonical 3-tuple from the directory.
    * ``--returns`` + ``--weights`` — explicit file paths. ``--dates`` is
      optional but recommended for real date ranges.
    """
    if args.results_dir is not None:
        rd = Path(args.results_dir)
        returns_path = rd / "portfolio_returns.npy"
        weights_path = rd / "weights.npy"
        dates_path = rd / "dates.npy"
        if not returns_path.exists():
            raise FileNotFoundError(
                f"{returns_path} not found — the run must have been "
                f"produced by the updated logger which saves canonical "
                f"artifact filenames."
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"{weights_path} not found — cannot compute regime turnover."
            )
        returns = np.load(returns_path)
        weights = np.load(weights_path)
        dates = np.load(dates_path) if dates_path.exists() else None
        return returns, weights, dates

    if args.returns is None:
        raise ValueError(
            "Must provide either --results_dir or --returns + --weights."
        )
    returns = np.load(args.returns)
    weights = (
        np.load(args.weights)
        if args.weights
        else np.ones((len(returns), 8)) / 8
    )
    dates = np.load(args.dates) if args.dates else None
    return returns, weights, dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-split analysis.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory holding weights.npy, portfolio_returns.npy, "
                             "and optionally dates.npy (preferred).")
    parser.add_argument("--returns", type=str, default=None,
                        help="Path to portfolio_returns .npy file.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights .npy file.")
    parser.add_argument("--dates", type=str, default=None,
                        help="Path to dates .npy file (int64 ns). Optional.")
    parser.add_argument("--start_date", type=str, default="2008-01-01",
                        help="Calendar date for index 0 when --dates is absent.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path. Defaults to "
                             "<results_dir>/regime_metrics.csv when possible.")
    args = parser.parse_args()

    returns, weights, dates = _resolve_inputs(args)

    if args.output is None:
        if args.results_dir is not None:
            output = Path(args.results_dir) / "regime_metrics.csv"
        else:
            output = Path("results/regime_metrics.csv")
    else:
        output = Path(args.output)

    rows = compute_regime_metrics(
        portfolio_returns=returns,
        weights=weights,
        dates=dates,
        start_date=args.start_date,
    )
    write_regime_csv(rows, output)

    print(f"\n{'Regime':<20} {'Days':>6} {'AnnRet':>10} {'Sharpe':>10} {'MDD':>10}")
    print("-" * 60)
    for r in rows:
        print(f"{r['regime']:<20} {r['n_days']:>6} {r['annual_return']:>+10.4f} "
              f"{r['sharpe_ratio']:>10.4f} {r['max_drawdown']:>10.4f}")


if __name__ == "__main__":
    main()
