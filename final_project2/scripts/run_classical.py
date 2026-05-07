"""
Classical portfolio benchmark evaluation.

Evaluates non-RL baselines on the test split for comparison with RL agents.

Usage:
    uv run src/scripts/run_classical.py --run_group=baselines --seed=0
    uv run src/scripts/run_classical.py --run_group=baselines --use_mutual_funds --start_date=1995-01-01 --seed=0
"""
import argparse
import os
import random
import numpy as np
import torch
import wandb
from scipy.optimize import minimize

from offline_rl.envs.data_utils import make_train_val_test_envs, DEFAULT_TICKERS


class EqualWeightStrategy:
    """1/N portfolio: w_i = 1/N for all assets."""

    def __init__(self, n_assets):
        self.n_assets = n_assets
        self.weights = np.ones(n_assets, dtype=np.float32) / n_assets

    def get_action(self, obs):
        return self.weights.copy()


class InverseVolatilityStrategy:
    """
    w_i proportional to 1/sigma_i.
    sigma_i extracted from rolling_std in the observation vector.
    obs layout: [weights(n), features(n*6)] where feature index 2 = rolling_std.
    """

    def __init__(self, n_assets):
        self.n_assets = n_assets

    def get_action(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy().flatten()
        # Extract rolling_std (index 2 in each asset's 6 features)
        rolling_std = np.array([
            obs[self.n_assets + i * 6 + 2] for i in range(self.n_assets)
        ])
        rolling_std = np.clip(np.abs(rolling_std), 1e-7, None)
        inv_vol = 1.0 / rolling_std
        weights = inv_vol / inv_vol.sum()
        return weights.astype(np.float32)


class MarkowitzMVOStrategy:
    """
    Markowitz Mean-Variance Optimization.
    w* = argmax (mu^T w - lambda/2 w^T Sigma w), s.t. sum(w)=1, w>=0

    Fit once on train split returns (static weights).
    """

    def __init__(self, returns_data: np.ndarray, risk_aversion: float = 1.0):
        self.n_assets = returns_data.shape[1]
        self.mu = returns_data.mean(axis=0)
        self.Sigma = np.cov(returns_data.T)
        self.risk_aversion = risk_aversion
        self._solve()

    def _solve(self):
        n = self.n_assets

        def objective(w):
            return -(self.mu @ w - 0.5 * self.risk_aversion * w @ self.Sigma @ w)

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n
        x0 = np.ones(n) / n
        result = minimize(objective, x0, bounds=bounds, constraints=constraints,
                          method='SLSQP')
        self.weights = result.x.astype(np.float32)

    def get_action(self, obs):
        return self.weights.copy()


def evaluate_strategy(strategy, env, n_episodes):
    """Run a strategy on an environment and collect metrics."""
    episode_returns, portfolio_values, turnovers = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_turnover = 0.0
        info = {}

        while not done:
            action = strategy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_turnover += info.get('turnover', 0.0)

        episode_returns.append(ep_return)
        portfolio_values.append(info.get('portfolio_value', 1.0))
        turnovers.append(ep_turnover)

    annual_returns = [(pv - 1.0) * (252 / env.episode_length) for pv in portfolio_values]
    return {
        'episode_return': float(np.mean(episode_returns)),
        'portfolio_value': float(np.mean(portfolio_values)),
        'annual_return': float(np.mean(annual_returns)),
        'avg_turnover': float(np.mean(turnovers)),
        'std_episode_return': float(np.std(episode_returns)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Classical portfolio benchmarks")
    parser.add_argument("--run_group", type=str, default="debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--use_mutual_funds", action="store_true")
    parser.add_argument("--start_date", type=str, default="2008-01-01")
    parser.add_argument("--train_end", type=str, default="2020-12-31")
    parser.add_argument("--val_start", type=str, default="2021-01-01")
    parser.add_argument("--val_end", type=str, default="2021-12-31")
    parser.add_argument("--test_start", type=str, default="2022-01-01")
    parser.add_argument("--end_date", type=str, default="2026-03-31")
    parser.add_argument("--episode_length", type=int, default=63)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--reward_type", type=str, default="log_return")
    parser.add_argument("--use_macro", action="store_true")
    parser.add_argument("--use_sentiment", action="store_true")
    parser.add_argument("--use_alpaca_embeddings", action="store_true")
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--risk_aversion", type=float, default=1.0,
                        help="Risk aversion parameter for Markowitz MVO")
    parser.add_argument("--save_results", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create environments. Classical baselines (equal-weight, inverse-vol,
    # Markowitz) all emit portfolio weights directly on the simplex, so the
    # env must not apply another softmax to them.
    train_env, val_env, test_env, metadata = make_train_val_test_envs(
        use_mutual_funds=args.use_mutual_funds,
        tickers=args.tickers,
        train_start=args.start_date,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.end_date,
        episode_length=args.episode_length,
        transaction_cost=args.transaction_cost,
        reward_type=args.reward_type,
        use_macro=args.use_macro,
        use_sentiment=args.use_sentiment,
        use_alpaca_embeddings=args.use_alpaca_embeddings,
        accept_portfolio_weights=True,
        fred_api_key=os.environ.get("FRED_API_KEY"),
    )

    n_assets = train_env.action_space.shape[0]
    print(f"Assets: {n_assets}")
    print(f"Train: {metadata['train_start']} -> {metadata['train_end']}")
    print(f"Test:  {metadata['test_start']} -> {metadata['test_end']}")

    # Init WandB
    wandb.init(
        project="cs285-portfolio-rl",
        group=args.run_group,
        name=f"classical_seed{args.seed}",
        config={**vars(args), **metadata},
    )

    # Build strategies
    strategies = {
        "equal_weight": EqualWeightStrategy(n_assets),
        "inverse_volatility": InverseVolatilityStrategy(n_assets),
        "markowitz_mvo": MarkowitzMVOStrategy(
            train_env.price_returns, risk_aversion=args.risk_aversion
        ),
    }

    # Evaluate each strategy on val and test
    all_results = {}
    for name, strategy in strategies.items():
        print(f"\nEvaluating {name}...")

        if hasattr(strategy, 'weights'):
            print(f"  Weights: {np.round(strategy.weights, 4)}")

        val_metrics = evaluate_strategy(strategy, val_env, args.n_eval_episodes)
        test_metrics = evaluate_strategy(strategy, test_env, args.n_eval_episodes)

        wandb.log({f"{name}/val/{k}": v for k, v in val_metrics.items()})
        wandb.log({f"{name}/test/{k}": v for k, v in test_metrics.items()})

        all_results[name] = {"val": val_metrics, "test": test_metrics}
        print(f"  Val:  return={val_metrics['episode_return']:.4f}, "
              f"PV={val_metrics['portfolio_value']:.4f}")
        print(f"  Test: return={test_metrics['episode_return']:.4f}, "
              f"PV={test_metrics['portfolio_value']:.4f}")

    # Save to CSV
    if args.save_results:
        import pandas as pd
        os.makedirs("results", exist_ok=True)
        rows = []
        for name, splits in all_results.items():
            for split, metrics in splits.items():
                rows.append({"strategy": name, "split": split, **metrics})
        df = pd.DataFrame(rows)
        path = f"results/classical_benchmarks_seed{args.seed}.csv"
        df.to_csv(path, index=False)
        print(f"\nSaved results to {path}")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
