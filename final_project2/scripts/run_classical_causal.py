"""Classical baselines (EW, Momentum, Risk-Parity) on the causal-pipeline
test window. Single chronological sweep, full 1,061-day test split.

Output: results/classical_causal.csv with columns
    strategy, sharpe_ratio, cumulative_return, max_drawdown,
    sum_turnover, avg_daily_turnover, annual_return, annual_volatility,
    n_steps

These rows are reference points for the main results table; the leak
fix in core/envs/data_utils.py:compute_features ensures the env is
on the causal pipeline (matches Phase 2C/2A-causal/Phase 2D).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.envs.data_utils import make_train_val_test_envs


TRADING_DAYS = 252
LOOKBACK = 60  # days, for momentum and risk-parity windows


def _equal_weight(n_assets: int, t: int, fwd_history: np.ndarray) -> np.ndarray:
    """1/N constant allocation."""
    return np.ones(n_assets, dtype=np.float64) / n_assets


def _momentum(n_assets: int, t: int, fwd_history: np.ndarray) -> np.ndarray:
    """Cumulative-return momentum, softmax-normalized.

    Uses last LOOKBACK rows of fwd_history (causally available), computes
    sum of log-returns per asset, applies softmax to get simplex weights.
    Falls back to EW if insufficient history.
    """
    if t < LOOKBACK:
        return _equal_weight(n_assets, t, fwd_history)
    window = np.log1p(np.clip(fwd_history[t - LOOKBACK:t], -0.9999, None))
    scores = window.sum(axis=0)
    e = np.exp(scores - scores.max())
    return e / e.sum()


def _risk_parity(n_assets: int, t: int, fwd_history: np.ndarray) -> np.ndarray:
    """Inverse-volatility weights (simplified risk-parity).

    Uses last LOOKBACK rows, computes per-asset std, allocates inversely.
    Falls back to EW if insufficient history or all stds near zero.
    """
    if t < LOOKBACK:
        return _equal_weight(n_assets, t, fwd_history)
    window = fwd_history[t - LOOKBACK:t]
    std = np.clip(window.std(axis=0), 1e-7, None)
    inv = 1.0 / std
    return inv / inv.sum()


def evaluate(strategy_fn, test_env) -> dict:
    """Single chronological sweep of test_env using strategy_fn(n_assets, t, fwd_history).

    fwd_history is the env's internal _forward_returns array sliced to
    the test split, made available so causal policies can compute
    momentum/vol windows. The strategy is responsible for reading only
    fwd_history[:t] (no lookahead).
    """
    obs, _ = test_env.reset(options={"randomize": False})
    n_assets = test_env.action_space.shape[0]
    fwd_history = test_env._forward_returns  # (T, N)

    weights_list = []
    simple_returns = []
    turnovers = []
    pv = [1.0]
    info = {}

    t = 0
    done = False
    while not done:
        w = strategy_fn(n_assets, t, fwd_history)
        obs, _, terminated, truncated, info = test_env.step(w)
        done = terminated or truncated
        weights_list.append(np.asarray(info["executed_weights"], dtype=np.float64))
        simple_returns.append(float(info["portfolio_return"]))
        turnovers.append(float(info["turnover"]))
        pv.append(float(info["portfolio_value"]))
        t += 1

    weights = np.array(weights_list)        # (T, N)
    rets = np.array(simple_returns)         # (T,)
    turn = np.array(turnovers)              # (T,)
    pv_arr = np.array(pv)                   # (T+1,)

    sharpe = (
        float(rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if rets.size > 1 and rets.std(ddof=1) > 1e-12 else 0.0
    )
    cum_ret = float(pv_arr[-1] - 1.0)
    running_max = np.maximum.accumulate(pv_arr)
    dd = (running_max - pv_arr) / np.maximum(running_max, 1e-12)
    max_dd = float(dd.max())
    sum_turnover = float(turn.sum())
    avg_daily_turnover = float(turn.mean())
    annual_return = float(rets.mean() * TRADING_DAYS)
    annual_volatility = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "sharpe_ratio": sharpe,
        "cumulative_return": cum_ret,
        "max_drawdown": max_dd,
        "sum_turnover": sum_turnover,
        "avg_daily_turnover": avg_daily_turnover,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "n_steps": int(rets.size),
    }


def main() -> int:
    # Build test env on the causal pipeline (post-leak-fix).
    _train, _val, test_env, metadata = make_train_val_test_envs(
        train_start="2008-01-01", train_end="2020-12-31",
        val_start="2021-01-01",   val_end="2021-12-31",
        test_start="2022-01-01",  test_end="2026-03-31",
        episode_length=63,
        transaction_cost=0.001,
        accept_portfolio_weights=True,
        fred_api_key=os.environ.get("FRED_API_KEY"),
    )
    # Use full test sweep, not 63-day windows.
    test_env.episode_length = metadata["T_test"]
    print(f"Test env: {metadata['test_start']} -> {metadata['test_end']}  "
          f"({metadata['T_test']} days)")

    strategies = {
        "equal_weight": _equal_weight,
        "momentum_60d": _momentum,
        "risk_parity_60d": _risk_parity,
    }

    rows = []
    for name, fn in strategies.items():
        # Reset between strategies to start fresh.
        m = evaluate(fn, test_env)
        m_row = {"strategy": name, **m}
        rows.append(m_row)
        print(
            f"{name:18s}  sharpe={m['sharpe_ratio']:+.4f}  "
            f"cum={m['cumulative_return']:+.4f}  "
            f"maxdd={m['max_drawdown']:.4f}  "
            f"avg_turn={m['avg_daily_turnover']:.4f}  "
            f"ann_ret={m['annual_return']:+.4f}  "
            f"steps={m['n_steps']}"
        )

    out_path = Path("results/classical_causal.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
