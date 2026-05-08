"""
Training entry point for SB3-backed online RL baselines.

Algorithms: A2C, PPO (SB3), SAC, TD3, DDPG, TQC (sb3-contrib)

DQN / HER are intentionally excluded:
  - DQN requires a Discrete action space; portfolio allocation is n_assets-dimensional continuous.
  - HER requires a GoalEnv interface (desired_goal / achieved_goal); not applicable here.

Usage:
    # On-policy (no action space wrapping needed)
    uv run src/scripts/run_sb3.py --base_config=a2c   --run_group=sb3_baselines --seed=0
    uv run src/scripts/run_sb3.py --base_config=ppo_sb3 --run_group=sb3_baselines --seed=0

    # Off-policy (ActionBoundedWrapper applied automatically)
    uv run src/scripts/run_sb3.py --base_config=sac_sb3 --run_group=sb3_baselines --seed=0
    uv run src/scripts/run_sb3.py --base_config=td3     --run_group=sb3_baselines --seed=0
    uv run src/scripts/run_sb3.py --base_config=ddpg    --run_group=sb3_baselines --seed=0
    uv run src/scripts/run_sb3.py --base_config=tqc     --run_group=sb3_baselines --seed=0

    # With FinRL environment (richer per-asset features)
    uv run src/scripts/run_sb3.py --base_config=td3 --use_finrl --run_group=sb3_finrl --seed=0

    # With macro / sentiment features
    uv run src/scripts/run_sb3.py --base_config=sac_sb3 --use_macro --run_group=sb3_macro --seed=0
"""
import argparse
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.envs.data_utils import make_train_test_envs, make_train_test_envs_finrl, DEFAULT_TICKERS
from online_rl.agents.sb3_agent import make_sb3_model, PortfolioEvalCallback, BOUNDED_ALGOS
from offline_rl.configs import CONFIG_MAP

SB3_ALGOS = {"a2c", "ppo_sb3", "sac_sb3", "td3", "ddpg", "tqc"}


def parse_args():
    parser = argparse.ArgumentParser(description="SB3 baseline training for portfolio RL")
    parser.add_argument("--run_group", type=str, default="debug")
    parser.add_argument(
        "--base_config", type=str, default="a2c",
        choices=sorted(SB3_ALGOS),
        help="Algorithm to run. DQN/HER are not applicable (see module docstring).",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Environment
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start_date", type=str, default="2008-01-01")
    parser.add_argument("--end_date", type=str, default="2026-03-31")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--episode_length", type=int, default=63)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument(
        "--reward_type", type=str, default="log_return",
        choices=["log_return", "diff_sharpe"],
    )
    parser.add_argument(
        "--use_finrl", action="store_true",
        help="Use FinRL's PortfolioOptimizationEnv (richer features: MACD, RSI, CCI, turbulence).",
    )
    parser.add_argument("--finrl_time_window", type=int, default=20)
    parser.add_argument(
        "--use_macro", action="store_true",
        help="Append 8 FRED macro features. Requires FRED_API_KEY env var.",
    )
    parser.add_argument(
        "--use_sentiment", action="store_true",
        help="Append SF Fed Daily News Sentiment Index.",
    )

    # Training
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)

    # Model save
    parser.add_argument(
        "--save_model", action="store_true",
        help="Save final model checkpoint to results/rl/<run_name>/model.zip",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.base_config not in SB3_ALGOS:
        raise ValueError(
            f"'{args.base_config}' is not a supported SB3 algorithm.\n"
            "DQN requires Discrete actions; HER requires GoalEnv — neither suits portfolio optimization.\n"
            f"Valid choices: {sorted(SB3_ALGOS)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build environments ──────────────────────────────────────────────────────
    tickers = args.tickers or DEFAULT_TICKERS
    print(f"Downloading market data: {tickers}")

    if args.use_finrl:
        if args.use_macro:
            print("  WARNING: --use_macro is ignored with --use_finrl. "
                  "FinRL uses its own feature pipeline (MACD, RSI, CCI, turbulence).")
        if args.use_sentiment:
            print("  WARNING: --use_sentiment is ignored with --use_finrl. "
                  "FinRL uses its own feature pipeline.")
        # For off-policy algos that need bounded actions, accept_portfolio_weights=False
        # means SB3 receives raw logits; FinRL normalizes internally via softmax.
        train_env, test_env, metadata = make_train_test_envs_finrl(
            tickers=tickers,
            start=args.start_date,
            end=args.end_date,
            train_ratio=args.train_ratio,
            time_window=args.finrl_time_window,
            transaction_cost=args.transaction_cost,
            accept_portfolio_weights=False,
            reward_type=args.reward_type,
        )
    else:
        train_env, test_env, metadata = make_train_test_envs(
            tickers=tickers,
            start=args.start_date,
            end=args.end_date,
            train_ratio=args.train_ratio,
            episode_length=args.episode_length,
            transaction_cost=args.transaction_cost,
            reward_type=args.reward_type,
            use_macro=args.use_macro,
            use_sentiment=args.use_sentiment,
            fred_api_key=os.environ.get("FRED_API_KEY"),
        )

    print(f"Train: {metadata['train_start']} → {metadata['train_end']}  ({metadata['T_train']} days)")
    print(f"Test:  {metadata['test_start']} → {metadata['test_end']}  ({metadata['T_test']} days)")
    print(f"obs_dim={metadata['obs_dim']}")

    # SB3 2.x requires finite action space bounds for all continuous-action algos
    from core.envs.action_bounded_wrapper import ActionBoundedWrapper
    train_env = ActionBoundedWrapper(train_env)
    test_env  = ActionBoundedWrapper(test_env)
    print(f"ActionBoundedWrapper applied (action space: {train_env.action_space})")

    # ── Load config ─────────────────────────────────────────────────────────────
    config_module = importlib.import_module(CONFIG_MAP[args.base_config])
    config = config_module.get_config()

    # ── WandB init ───────────────────────────────────────────────────────────────
    run_name = f"{args.base_config}_seed{args.seed}"
    wandb.init(
        project="cs285-portfolio-rl",
        group=args.run_group,
        name=run_name,
        config={**dict(config), **vars(args), **metadata},
        sync_tensorboard=True,  # SB3 logs to TensorBoard; WandbCallback syncs it
    )

    # ── Build model ──────────────────────────────────────────────────────────────
    model = make_sb3_model(args.base_config, train_env, config, device=device)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Algorithm: {args.base_config} | Parameters: {n_params:,}")

    # ── Callbacks ────────────────────────────────────────────────────────────────
    wandb_cb = WandbCallback(
        gradient_save_freq=0,   # don't save gradient histograms (slow)
        verbose=0,
    )
    eval_cb = PortfolioEvalCallback(
        eval_env=test_env,
        eval_interval=args.eval_interval,
        n_eval_episodes=args.n_eval_episodes,
        verbose=1,
    )

    # ── Training ─────────────────────────────────────────────────────────────────
    print(f"\nTraining {args.base_config} for {args.total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[wandb_cb, eval_cb],
        progress_bar=True,
    )

    # ── Final evaluation ─────────────────────────────────────────────────────────
    print("\nFinal evaluation (20 episodes)...")
    orig_n = eval_cb.n_eval_episodes
    eval_cb.n_eval_episodes = 20
    final_metrics = eval_cb._run_eval()
    eval_cb.n_eval_episodes = orig_n

    wandb.log({f"final/{k.replace('eval/', '')}": v for k, v in final_metrics.items()})
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Save model ───────────────────────────────────────────────────────────────
    if args.save_model:
        save_dir = f"results/rl/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        model.save(f"{save_dir}/model")
        print(f"Model saved to {save_dir}/model.zip")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
