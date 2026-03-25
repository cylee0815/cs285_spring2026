"""
Training entry point for online RL portfolio agents.

Usage:
    uv run src/scripts/run.py --run_group=debug --base_config=ppo --seed=0
    uv run src/scripts/run.py --run_group=exp1 --base_config=ppo_lstm --reward_type=diff_sharpe
    uv run src/scripts/run.py --run_group=exp1 --base_config=ppo_transformer --seed=0
"""
import argparse
import importlib
import random
import numpy as np
import torch
import wandb
from tqdm import trange

from src.envs.data_utils import make_train_test_envs, DEFAULT_TICKERS
from src.agents.ppo import PPOAgent
from src.configs import CONFIG_MAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", type=str, default="debug")
    parser.add_argument(
        "--base_config", type=str, default="ppo", choices=list(CONFIG_MAP.keys())
    )
    parser.add_argument("--seed", type=int, default=0)

    # Environment args
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Asset tickers (default: ETF universe)",
    )
    parser.add_argument("--start_date", type=str, default="2010-01-01")
    parser.add_argument("--end_date", type=str, default="2024-01-01")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--episode_length", type=int, default=63)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument(
        "--reward_type",
        type=str,
        default="log_return",
        choices=["log_return", "diff_sharpe"],
    )

    # Training args
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)

    # Config overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config_module = importlib.import_module(CONFIG_MAP[args.base_config])
    config = config_module.get_config()

    # Apply CLI overrides
    for key in ["lr", "n_steps", "batch_size"]:
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    # Build environments
    tickers = args.tickers or DEFAULT_TICKERS
    print(f"Downloading market data for: {tickers}")
    train_env, test_env, metadata = make_train_test_envs(
        tickers=tickers,
        start=args.start_date,
        end=args.end_date,
        train_ratio=args.train_ratio,
        episode_length=args.episode_length,
        transaction_cost=args.transaction_cost,
        reward_type=args.reward_type,
    )
    print(
        f"Train: {metadata['train_start']} to {metadata['train_end']}"
        f" ({metadata['T_train']} days)"
    )
    print(
        f"Test:  {metadata['test_start']} to {metadata['test_end']}"
        f" ({metadata['T_test']} days)"
    )

    # Initialize WandB
    run_name = f"{args.base_config}_seed{args.seed}"
    wandb.init(
        project="cs285-portfolio-rl",
        group=args.run_group,
        name=run_name,
        config={
            **dict(config),
            **vars(args),
            **metadata,
        },
    )

    # Build agent
    agent = PPOAgent(train_env, config, device)
    n_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"Policy: {args.base_config} | Parameters: {n_params:,}")

    # Training loop
    timesteps = 0
    update_count = 0

    with trange(0, args.total_timesteps, config.n_steps, desc="Training") as pbar:
        for _ in pbar:
            metrics = agent.update()
            timesteps += config.n_steps
            update_count += 1

            # Log to WandB
            log_dict = {f"train/{k}": v for k, v in metrics.items()}
            log_dict["timesteps"] = timesteps

            # Periodic evaluation
            if timesteps % args.eval_interval < config.n_steps:
                eval_metrics = agent.evaluate(
                    test_env, n_episodes=args.n_eval_episodes
                )
                log_dict.update(eval_metrics)
                pbar.set_postfix(
                    {
                        "annual_ret": f"{eval_metrics.get('eval/annual_return', 0):.2%}",
                        "ep_ret": f"{metrics.get('episode_return', 0):.4f}",
                    }
                )

            wandb.log(log_dict, step=timesteps)

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = agent.evaluate(test_env, n_episodes=20)
    wandb.log(
        {f"final/{k.replace('eval/', '')}": v for k, v in final_metrics.items()}
    )

    print("Training complete!")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
