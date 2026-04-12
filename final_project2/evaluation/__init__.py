"""Walk-forward backtesting, metrics, and classical baselines.

* ``metrics.py`` — Sharpe, max drawdown, turnover, cumulative return, etc.
* ``baselines.py`` — equal weight, momentum, risk parity, buy-and-hold.
* ``backtest.py`` — walk-forward backtest runner for trained IQL agents.

Submodules are imported lazily to avoid pulling in ``torch`` when only
metrics or baselines are needed.
"""
