"""CLI entrypoint for offline dataset generation.

Usage:
    python scripts/build_dataset.py \\
        --policy dirichlet \\
        --seed 42 \\
        --output datasets/dirichlet_dataset.npz

    python scripts/build_dataset.py \\
        --policy equal_weight \\
        --seed 0 \\
        --output datasets/ew_dataset.npz \\
        --config experiments/configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.build_dataset import build_dataset_from_env, save_dataset
from env.portfolio_env import PortfolioEnv
from policies.behavior import (
    DirichletPolicy,
    EqualWeightPolicy,
    MomentumPolicy,
    RiskParityPolicy,
)


POLICY_NAMES = ["equal_weight", "dirichlet", "momentum", "risk_parity"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an offline RL dataset using a behavior policy.",
    )
    parser.add_argument(
        "--policy",
        required=True,
        choices=POLICY_NAMES,
        help="Behavior policy to use for data generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/dataset.npz",
        help="Output path for the .npz dataset file.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="Number of timesteps in the synthetic environment.",
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=8,
        help="Number of assets.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost lambda.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=1.0,
        help="Dirichlet concentration parameter (only for dirichlet policy).",
    )
    parser.add_argument(
        "--momentum-lookback",
        type=int,
        default=60,
        help="Lookback window for momentum policy.",
    )
    parser.add_argument(
        "--risk-parity-lookback",
        type=int,
        default=60,
        help="Lookback window for risk parity policy.",
    )
    return parser


def _make_policy(
    name: str,
    n_assets: int,
    seed: int,
    dirichlet_alpha: float = 1.0,
    momentum_lookback: int = 60,
    risk_parity_lookback: int = 60,
):
    """Instantiate a behavior policy by name."""
    if name == "equal_weight":
        return EqualWeightPolicy(n_assets=n_assets)
    elif name == "dirichlet":
        return DirichletPolicy(n_assets=n_assets, alpha=dirichlet_alpha, seed=seed)
    elif name == "momentum":
        return MomentumPolicy(n_assets=n_assets, lookback=momentum_lookback)
    elif name == "risk_parity":
        return RiskParityPolicy(n_assets=n_assets, lookback=risk_parity_lookback)
    else:
        raise ValueError(f"Unknown policy: {name}")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    rng = np.random.RandomState(args.seed)

    # Generate synthetic features and returns
    n_features = 10
    features = rng.randn(args.n_steps, n_features).astype(np.float32)
    forward_returns = rng.randn(args.n_steps, args.n_assets).astype(np.float32) * 0.01

    env = PortfolioEnv(
        features=features,
        forward_returns=forward_returns,
        transaction_cost_lambda=args.transaction_cost,
    )

    policy = _make_policy(
        name=args.policy,
        n_assets=args.n_assets,
        seed=args.seed,
        dirichlet_alpha=args.dirichlet_alpha,
        momentum_lookback=args.momentum_lookback,
        risk_parity_lookback=args.risk_parity_lookback,
    )

    dataset = build_dataset_from_env(env, policy, seed=args.seed)

    save_dataset(
        dataset,
        args.output,
        metadata={
            "n_assets": args.n_assets,
            "feature_dim": n_features,
            "seed": args.seed,
            "policy_name": args.policy,
        },
    )

    print(
        f"Dataset saved to {args.output}: "
        f"{len(dataset['states'])} transitions, "
        f"policy={args.policy}, seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
