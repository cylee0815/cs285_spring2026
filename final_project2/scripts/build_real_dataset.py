"""Build an offline RL dataset from real historical market data.

Downloads (or reads from cache) adjusted-close prices for the 8-asset ETF
basket, computes causal features, rolls out a behaviour policy through a
``PortfolioEnv`` built on the real forward returns, and saves the resulting
transitions — *together with each transition's calendar date* — as an
``.npz`` file.

The date array is what lets :mod:`data.splits` partition the dataset into
strictly time-ordered train / validation / test subsets.

Warm-up isolation
-----------------
Feature engineering needs ~6 months of history to fill rolling windows
(longest default window = 126-day momentum). To ensure the first dated
transition in the dataset coincides with the user's intended evaluation
start (not with the warmup-trim point), we download from
``--download-start`` and then trim the feature bundle to
``--eval-start`` after all rolling windows have fully warmed up.

    download_start  ──►  warmup window (~6 months)  ──►  eval_start
                                                           (first row
                                                            in dataset)

Usage
-----
    uv run python scripts/build_real_dataset.py \\
        --policy dirichlet \\
        --output datasets/real_dirichlet.npz \\
        --download-start 2007-03-01 \\
        --eval-start 2008-01-01 \\
        --end 2026-03-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.build_dataset import build_dataset_from_env, save_dataset
from data.download_data import download_prices
from env.portfolio_env import PortfolioEnv
from features.feature_engineering import FeatureBundle, build_features
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
    parser.add_argument(
        "--download-start",
        default="2007-03-01",
        help=(
            "Inclusive first date fetched from Yahoo. Must precede --eval-start "
            "by enough to fill all rolling-feature windows (≥ ~6 months for "
            "the default momentum/MA windows)."
        ),
    )
    parser.add_argument(
        "--eval-start",
        default="2008-01-01",
        help=(
            "Inclusive first date that appears in the saved dataset. "
            "Features between --download-start and --eval-start exist only "
            "to provide warmup history for the rolling windows; those rows "
            "are dropped before any transitions are written."
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        help=argparse.SUPPRESS,  # deprecated alias for --eval-start
    )
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--cache-dir", default=".cache/prices")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "Ignore any cached parquet files and re-download prices. Use this "
            "when extending the date range backward past the cache's earliest "
            "date (otherwise download_prices() silently reuses the narrower "
            "cache window)."
        ),
    )
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

    # Back-compat: --start used to be the download start; it is now an alias
    # for --eval-start. Explicit --eval-start/--download-start take precedence.
    if args.start is not None:
        print("[warn] --start is deprecated; use --eval-start (and optionally "
              "--download-start for the warmup buffer).")
        args.eval_start = args.start

    download_start = pd.Timestamp(args.download_start)
    eval_start = pd.Timestamp(args.eval_start)
    end_ts = pd.Timestamp(args.end)
    if download_start > eval_start:
        raise ValueError(
            f"--download-start ({args.download_start}) must be <= "
            f"--eval-start ({args.eval_start}) so warmup data precedes "
            f"the first dated transition."
        )
    if eval_start >= end_ts:
        raise ValueError(
            f"--eval-start ({args.eval_start}) must precede --end ({args.end})."
        )
    warmup_days = (eval_start - download_start).days
    if warmup_days < 180:
        print(f"[warn] warmup buffer is only {warmup_days} days; rolling "
              "features with windows ≥ that size may eat into the evaluation "
              "range. Recommend ≥ 180 calendar days.")

    prices = download_prices(
        tickers=list(args.tickers),
        start=args.download_start,
        end=args.end,
        cache_dir=args.cache_dir,
        use_cache=not args.force_download,
    )
    print(f"Loaded prices: {prices.shape[0]} rows × {prices.shape[1]} tickers "
          f"({prices.index.min().date()} → {prices.index.max().date()})")
    # The invariant we actually care about is: there is enough price
    # history before eval_start for every rolling window to fully warm up.
    # Raw gap vs. --download-start is uninformative because some tickers
    # have later inception dates (e.g., HYG launched 2007-04-11), which
    # no amount of cache busting can fix.
    effective_warmup = (eval_start - prices.index.min()).days
    longest_window = max(
        [args.lookback_returns, args.volatility_window, *args.ma_windows]
        + ([] if args.no_momentum else list(args.momentum_windows))
    )
    # Rolling windows are in *trading* days but we measure in calendar
    # days; 1.6× is a conservative inflation factor (~252/365 inverse plus
    # slack).
    min_warmup_cal_days = int(np.ceil(longest_window * 1.6))
    if effective_warmup < min_warmup_cal_days:
        raise RuntimeError(
            f"Effective warmup buffer is only {effective_warmup} calendar "
            f"days ({prices.index.min().date()} → {eval_start.date()}), but "
            f"the longest rolling window is {longest_window} trading days "
            f"(≈{min_warmup_cal_days} calendar days needed). Rolling "
            "features would truncate into the evaluation range. Fixes: "
            "move --eval-start later, move --download-start earlier "
            "(if data exists), or shorten the longest momentum/MA window."
        )
    first_gap_days = (prices.index.min() - download_start).days
    if first_gap_days > 30 and not args.force_download:
        # Most likely cause: stale cache narrower than the requested window.
        # (If it's real ticker inception — e.g., HYG — --force-download
        # will confirm by producing the same gap from a fresh fetch.)
        print(
            f"[warn] First available date {prices.index.min().date()} is "
            f"{first_gap_days} days after --download-start "
            f"{download_start.date()}. If you just shifted --download-start "
            "earlier, re-run with --force-download to refresh the cache. "
            "Otherwise this reflects real ticker inception; "
            f"effective warmup = {effective_warmup} days — OK."
        )

    bundle = build_features(
        prices,
        lookback_returns=args.lookback_returns,
        volatility_window=args.volatility_window,
        ma_windows=tuple(args.ma_windows),
        include_momentum=not args.no_momentum,
        momentum_windows=tuple(args.momentum_windows),
    )
    print(f"Features (pre-trim): {bundle.num_steps} rows × "
          f"{bundle.feature_dim} features; dates "
          f"{bundle.dates.min().date()} → {bundle.dates.max().date()}")

    # Drop the warmup prefix. After `build_features` the bundle's earliest
    # row is already NaN-free (rolling windows fully filled); here we
    # additionally require that row to fall on or after `eval_start` so the
    # first transition in the dataset is on the user-requested evaluation
    # boundary, regardless of how large the rolling windows were.
    bundle = _trim_bundle_to_start(bundle, eval_start)
    print(f"Features (post-trim @ {eval_start.date()}): {bundle.num_steps} "
          f"rows; first date {bundle.dates.min().date()}, "
          f"last date {bundle.dates.max().date()}")

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
            "download_start": args.download_start,
            "eval_start": args.eval_start,
            "start": args.eval_start,  # back-compat with existing readers
            "end": args.end,
        },
    )
    n_tr = len(dataset["states"])
    print(f"Saved {n_tr} transitions to {args.output}")
    return 0


def _trim_bundle_to_start(bundle: FeatureBundle, start: pd.Timestamp) -> FeatureBundle:
    """Drop warmup rows so bundle.dates[0] >= start.

    This is the final alignment step between the feature pipeline (which
    trims by warmup completion) and the calendar-date semantics expected by
    downstream splits. All four aligned arrays (features, forward_returns,
    dates) are sliced by the same integer offset so zero shift is
    introduced.
    """
    mask = np.asarray(bundle.dates >= start)
    offset = int(np.argmax(mask)) if mask.any() else len(bundle.dates)
    if offset >= len(bundle.dates):
        raise ValueError(
            f"All bundle dates precede start={start.date()} — nothing to keep."
        )
    if offset == 0:
        # Warmup already ends before `start`; no trim needed.
        return bundle
    return FeatureBundle(
        features=np.ascontiguousarray(bundle.features[offset:]),
        forward_returns=np.ascontiguousarray(bundle.forward_returns[offset:]),
        dates=bundle.dates[offset:],
        feature_names=bundle.feature_names,
        tickers=bundle.tickers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
