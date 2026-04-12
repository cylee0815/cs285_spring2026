"""Build an offline RL dataset from real historical market data.

Downloads (or reads from cache) adjusted-close prices for the 8-asset ETF
basket, computes causal features, rolls out a behaviour policy through a
``PortfolioEnv`` built on the real forward returns, and saves the resulting
transitions — *together with each transition's calendar date* — as an
``.npz`` file.

The date array is what lets :mod:`data.splits` partition the dataset into
strictly time-ordered train / validation / test subsets.

Usage
-----
    uv run python scripts/build_real_dataset.py \\
        --policy dirichlet \\
        --output datasets/real_dirichlet.npz \\
        --start 2008-01-01 --end 2026-03-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.build_dataset import build_dataset_from_env, save_dataset
from data.download_data import download_prices
from env.portfolio_env import PortfolioEnv
from features.feature_engineering import build_features
from policies.behavior import (
    DirichletPolicy,
    EqualWeightPolicy,
    MomentumPolicy,
    RiskParityPolicy,
)


DEFAULT_TICKERS = ["SPY", "EEM", "TLT", "HYG", "DBC", "GLD", "UUP", "SHY"]


def _make_policy(
    name: str,
    n_assets: int,
    seed: int,
    dirichlet_alpha: float,
    momentum_lookback: int,
    risk_parity_lookback: int,
):
    if name == "equal_weight":
        return EqualWeightPolicy(n_assets=n_assets)
    if name == "dirichlet":
        return DirichletPolicy(n_assets=n_assets, alpha=dirichlet_alpha, seed=seed)
    if name == "momentum":
        return MomentumPolicy(n_assets=n_assets, lookback=momentum_lookback)
    if name == "risk_parity":
        return RiskParityPolicy(n_assets=n_assets, lookback=risk_parity_lookback)
    raise ValueError(f"Unknown policy: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a date-aware offline RL dataset from real prices."
    )
    parser.add_argument("--policy", choices=[
        "dirichlet", "equal_weight", "momentum", "risk_parity",
    ], default="dirichlet")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--cache-dir", default=".cache/prices")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--momentum-lookback", type=int, default=60)
    parser.add_argument("--risk-parity-lookback", type=int, default=60)
    # Feature config — match experiments/configs/default.yaml.
    parser.add_argument("--lookback-returns", type=int, default=20)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--ma-windows", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--no-momentum", action="store_true")
    parser.add_argument(
        "--momentum-windows", nargs="+", type=int, default=[21, 63, 126]
    )
    args = parser.parse_args()

    prices = download_prices(
        tickers=list(args.tickers),
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        use_cache=True,
    )
    print(f"Loaded prices: {prices.shape[0]} rows × {prices.shape[1]} tickers "
          f"({prices.index.min().date()} → {prices.index.max().date()})")

    bundle = build_features(
        prices,
        lookback_returns=args.lookback_returns,
        volatility_window=args.volatility_window,
        ma_windows=tuple(args.ma_windows),
        include_momentum=not args.no_momentum,
        momentum_windows=tuple(args.momentum_windows),
    )
    print(f"Features: {bundle.num_steps} rows × {bundle.feature_dim} features; "
          f"{bundle.num_assets} assets; dates "
          f"{bundle.dates.min().date()} → {bundle.dates.max().date()}")

    env = PortfolioEnv(
        features=bundle.features,
        forward_returns=bundle.forward_returns,
        transaction_cost_lambda=args.transaction_cost,
    )

    policy = _make_policy(
        name=args.policy,
        n_assets=bundle.num_assets,
        seed=args.seed,
        dirichlet_alpha=args.dirichlet_alpha,
        momentum_lookback=args.momentum_lookback,
        risk_parity_lookback=args.risk_parity_lookback,
    )

    dataset = build_dataset_from_env(
        env, policy, seed=args.seed, dates=bundle.dates
    )
    # Persist the full forward-returns matrix alongside the transitions so
    # downstream backtests on val/test slices reconstruct rewards from the
    # same ground-truth prices the dataset was built from.
    dataset["forward_returns"] = bundle.forward_returns.astype(np.float32)

    save_dataset(
        dataset,
        args.output,
        metadata={
            "tickers": np.array(args.tickers),
            "n_assets": bundle.num_assets,
            "feature_dim": bundle.feature_dim,
            "policy_name": args.policy,
            "seed": args.seed,
            "transaction_cost": args.transaction_cost,
            "start": args.start,
            "end": args.end,
        },
    )
    n_tr = len(dataset["states"])
    print(f"Saved {n_tr} transitions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
