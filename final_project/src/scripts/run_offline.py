"""
Unified training script for offline RL baselines.

Supports: bc, fisher_bc, td3_bc, awac, cql_vanilla, iql, edac, bcq,
          mbpo, mopo, decision_transformer, trajectory_transformer

Usage:
    uv run src/scripts/run_offline.py --base_config=bc --run_group=debug --seed=0
    uv run src/scripts/run_offline.py --base_config=iql --run_group=debug --seed=0
    uv run src/scripts/run_offline.py --base_config=td3_bc --run_group=debug --n_offline_updates=1000 --seed=0
"""
import argparse
import importlib
import os
import random
import numpy as np
import torch
import wandb
from tqdm import trange

from src.envs.data_utils import (
    make_train_val_test_envs, DEFAULT_TICKERS, MUTUAL_FUND_TICKERS,
)
from src.agents.replay_buffer import ReplayBuffer, NStepReplayBuffer
from src.configs import CONFIG_MAP


AGENT_MAP = {
    "bc": ("src.agents.bc", "BCAgent"),
    "fisher_bc": ("src.agents.fisher_bc", "FisherBCAgent"),
    "td3_bc": ("src.agents.td3_bc", "TD3BCAgent"),
    "awac": ("src.agents.awac", "AWACAgent"),
    "cql_vanilla": ("src.agents.cql_vanilla", "VanillaCQLAgent"),
    "iql": ("src.agents.iql", "IQLAgent"),
    "edac": ("src.agents.edac", "EDACAgent"),
    "bcq": ("src.agents.bcq", "BCQAgent"),
    "mbpo": ("src.agents.mbpo", "MBPOAgent"),
    "mopo": ("src.agents.mopo", "MOPOAgent"),
    "decision_transformer": ("src.agents.decision_transformer", "DecisionTransformerAgent"),
    "trajectory_transformer": ("src.agents.trajectory_transformer", "TrajectoryTransformerAgent"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Offline RL baselines for portfolio optimization")
    parser.add_argument("--base_config", type=str, required=True, choices=list(AGENT_MAP.keys()))
    parser.add_argument("--run_group", type=str, default="debug")
    parser.add_argument("--seed", type=int, default=0)
    # Ticker universe
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--use_mutual_funds", action="store_true")
    # Chronological split
    parser.add_argument("--start_date", type=str, default="2008-01-01")
    parser.add_argument("--train_end", type=str, default="2020-12-31")
    parser.add_argument("--val_start", type=str, default="2021-01-01")
    parser.add_argument("--val_end", type=str, default="2021-12-31")
    parser.add_argument("--test_start", type=str, default="2022-01-01")
    parser.add_argument("--end_date", type=str, default="2026-03-31")
    parser.add_argument("--episode_length", type=int, default=63)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--reward_type", type=str, default="log_return",
                        choices=["log_return", "diff_sharpe"])
    # Feature flags
    parser.add_argument("--use_macro", action="store_true")
    parser.add_argument("--use_sentiment", action="store_true")
    parser.add_argument("--use_alpaca_embeddings", action="store_true")
    # Training
    parser.add_argument("--offline_data_steps", type=int, default=50_000)
    parser.add_argument("--n_offline_updates", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=5_000)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    # Multi-step returns
    parser.add_argument("--n_step", type=int, default=1,
                        help="N-step returns (1=standard, 3/5/10 for multi-step)")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_sharpe_ratio(step_returns: list, annualize: bool = True) -> float:
    """Compute Sharpe ratio from per-step returns."""
    if len(step_returns) < 2:
        return 0.0
    arr = np.array(step_returns)
    mean_r = arr.mean()
    std_r = arr.std()
    if std_r < 1e-8:
        return 0.0
    sharpe = mean_r / std_r
    if annualize:
        sharpe *= np.sqrt(252)
    return float(sharpe)


def compute_max_drawdown(portfolio_values: list) -> float:
    """Compute maximum drawdown from a sequence of portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0
    arr = np.array(portfolio_values)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (running_max - arr) / np.maximum(running_max, 1e-8)
    return float(drawdowns.max())


@torch.no_grad()
def evaluate_agent(agent, env, n_episodes, device):
    """Unified evaluation for all offline agents."""
    episode_returns, portfolio_values, turnovers = [], [], []
    all_sharpes, all_max_drawdowns = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_turnover = 0.0
        step_returns = []
        pv_trajectory = [1.0]
        info = {}

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.get_action(obs_t)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_turnover += info.get('turnover', 0.0)
            step_returns.append(reward)
            pv_trajectory.append(info.get('portfolio_value', pv_trajectory[-1]))

        episode_returns.append(ep_return)
        portfolio_values.append(info.get('portfolio_value', 1.0))
        turnovers.append(ep_turnover)
        all_sharpes.append(compute_sharpe_ratio(step_returns))
        all_max_drawdowns.append(compute_max_drawdown(pv_trajectory))

    annual_returns = [(pv - 1.0) * (252 / env.episode_length) for pv in portfolio_values]
    return {
        'eval/episode_return': float(np.mean(episode_returns)),
        'eval/portfolio_value': float(np.mean(portfolio_values)),
        'eval/annual_return': float(np.mean(annual_returns)),
        'eval/avg_turnover': float(np.mean(turnovers)),
        'eval/std_episode_return': float(np.std(episode_returns)),
        'eval/sharpe_ratio': float(np.mean(all_sharpes)),
        'eval/max_drawdown': float(np.mean(all_max_drawdowns)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create environments
    tickers = args.tickers
    ticker_label = (
        "mutual_fund_proxies" if (args.use_mutual_funds and tickers is None)
        else str(tickers or DEFAULT_TICKERS)
    )
    print(f"Downloading market data: {ticker_label}")

    train_env, val_env, test_env, metadata = make_train_val_test_envs(
        use_mutual_funds=args.use_mutual_funds,
        tickers=tickers,
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
        fred_api_key=os.environ.get("FRED_API_KEY"),
    )

    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    print(f"Train (offline): {metadata['train_start']} -> {metadata['train_end']}  ({metadata['T_train']} days)")
    print(f"Val   (HP eval): {metadata['val_start']} -> {metadata['val_end']}  ({metadata['T_val']} days)")
    print(f"Test  (eval):    {metadata['test_start']} -> {metadata['test_end']}  ({metadata['T_test']} days)")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")

    # Load config
    config_module = importlib.import_module(f"src.configs.{args.base_config}_config")
    config = config_module.get_config()
    if args.n_offline_updates:
        config.n_offline_updates = args.n_offline_updates
    if args.n_step > 1 and hasattr(config, 'gamma'):
        config.n_step = args.n_step

    # Init WandB
    run_name = f"{args.base_config}_seed{args.seed}"
    wandb.init(
        project="cs285-portfolio-rl",
        group=args.run_group,
        name=run_name,
        config={**dict(config), **vars(args), **metadata},
    )

    # Build offline buffer
    seq_len = getattr(config, 'regime_window', 20)
    if args.n_step > 1:
        gamma = getattr(config, 'gamma', 0.99)
        offline_buffer = NStepReplayBuffer(
            config.offline_buffer_size, obs_dim, action_dim, device,
            seq_len=seq_len, n_step=args.n_step, gamma=gamma,
        )
    else:
        offline_buffer = ReplayBuffer(
            config.offline_buffer_size, obs_dim, action_dim, device,
            seq_len=seq_len,
        )
    offline_buffer.load_from_env(train_env, n_steps=args.offline_data_steps)
    offline_buffer.freeze()

    # Build agent
    module_path, class_name = AGENT_MAP[args.base_config]
    agent_module = importlib.import_module(module_path)
    AgentClass = getattr(agent_module, class_name)
    agent = AgentClass(obs_dim, action_dim, config, device, offline_buffer=offline_buffer)

    # Training loop
    n_updates = config.n_offline_updates
    for step in trange(n_updates, desc=args.base_config):
        metrics = agent.update()
        if step % args.eval_interval == 0 and metrics:
            eval_metrics = evaluate_agent(agent, val_env, args.n_eval_episodes, device)
            wandb.log({
                **{f"train/{k}": v for k, v in metrics.items()},
                **eval_metrics,
                "step": step,
            })

    # Final evaluation on test set
    test_metrics = evaluate_agent(agent, test_env, args.n_eval_episodes * 2, device)
    wandb.log({f"test/{k.split('/')[-1]}": v for k, v in test_metrics.items()})
    print(f"Test results: {test_metrics}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
