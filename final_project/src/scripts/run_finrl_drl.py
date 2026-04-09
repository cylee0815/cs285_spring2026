"""
FinRL-native multi-algorithm comparison using DRLAgent.

Uses FinRL's DRLAgent class (finrl.agents.stablebaselines3.models.DRLAgent) as
the canonical FinRL interface for model creation.  All 5 algorithms supported by
FinRL's portfolio pipeline are trained sequentially and evaluated on the same
test environment, making head-to-head comparison straightforward.

Algorithms: PPO, A2C, DDPG, SAC, TD3
  (TQC is only in sb3-contrib, not FinRL's MODELS dict; use run_sb3.py for TQC)

Chronological split (aligned to the proposal's distribution-shift hypothesis):
  Train: 2008-01-01 → 2020-12-31  (GFC + COVID — offline pre-training regime)
  Val:   2021-01-01 → 2021-12-31  (used to pick best checkpoint per algo)
  Test:  2022-01-01 → 2026-03-31  (stock/bond correlation break — OOD regime)

Usage:
    # Train all 5 algorithms with FinRL env, single seed
    uv run src/scripts/run_finrl_drl.py --run_group=finrl_compare --seed=0

    # Train a single algorithm
    uv run src/scripts/run_finrl_drl.py --run_group=finrl_compare --algos ppo --seed=0

    # Use mutual fund proxies for extended history (1995–)
    uv run src/scripts/run_finrl_drl.py --run_group=finrl_mf --use_mutual_funds --start_date=1995-01-01 --seed=0

    # Add macro features
    uv run src/scripts/run_finrl_drl.py --run_group=finrl_macro --use_macro --seed=0
"""
import argparse
import importlib
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from src.envs.data_utils import DEFAULT_TICKERS, MUTUAL_FUND_TICKERS
from src.envs.finrl_wrapper import download_finrl_data, make_finrl_envs, FINRL_AVAILABLE
from src.agents.sb3_agent import PortfolioEvalCallback, BOUNDED_ALGOS


# ── FinRL DRLAgent import (avoids FinRL's broken __init__.py) ────────────────
def _import_drl_agent():
    try:
        _mod = importlib.import_module("finrl.agents.stablebaselines3.models")
        return _mod.DRLAgent
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "Could not import FinRL's DRLAgent. Make sure finrl is installed:\n"
            f"  uv sync\nOriginal error: {e}"
        )


# ── Per-algorithm SB3 constructor kwargs passed via DRLAgent.get_model ───────
# These mirror the hyperparameters in the individual *_config.py files but are
# expressed as flat dicts for DRLAgent's model_kwargs interface.
ALGO_MODEL_KWARGS = {
    "ppo": {
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "learning_rate": 3e-4,
    },
    "a2c": {
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "learning_rate": 7e-4,
    },
    "sac": {
        "buffer_size": 100_000,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 1_000,
    },
    "td3": {
        "buffer_size": 100_000,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 1_000,
        "policy_delay": 2,
        "action_noise": "NormalActionNoise",   # FinRL resolves this string to the class
    },
    "ddpg": {
        "buffer_size": 100_000,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 1_000,
        "action_noise": "OrnsteinUhlenbeckActionNoise",
    },
}

# SB3 2.x requires finite bounds for ALL continuous-action algos — always wrap
_BOUNDED = {"ppo", "a2c", "sac", "td3", "ddpg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FinRL DRLAgent multi-algorithm portfolio comparison"
    )
    parser.add_argument("--run_group", type=str, default="debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algos", nargs="+",
        default=["ppo", "a2c", "sac", "td3", "ddpg"],
        choices=list(ALGO_MODEL_KWARGS.keys()),
        help="Algorithms to train. Default: all 5.",
    )

    # Ticker universe
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument(
        "--use_mutual_funds", action="store_true",
        help="Use mutual fund proxies (VFINX, VUSTX, …) for extended history to 1990s.",
    )

    # Chronological splits (aligned to proposal)
    parser.add_argument("--start_date",  type=str, default="2008-01-01")
    parser.add_argument("--train_end",   type=str, default="2020-12-31")
    parser.add_argument("--val_start",   type=str, default="2021-01-01")
    parser.add_argument("--val_end",     type=str, default="2021-12-31")
    parser.add_argument("--test_start",  type=str, default="2022-01-01")
    parser.add_argument("--end_date",    type=str, default="2026-03-31")

    # FinRL env options
    parser.add_argument("--time_window", type=int, default=20,
                        help="FinRL obs lookback window in days.")
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument(
        "--use_macro", action="store_true",
        help="Append 8 FRED macro features. Requires FRED_API_KEY env var.",
    )

    # Training scale
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)

    # Output
    parser.add_argument(
        "--save_results", action="store_true",
        help="Save per-algo portfolio value CSVs to results/finrl_compare/",
    )
    parser.add_argument(
        "--save_models", action="store_true",
        help="Save trained model .zip files to results/finrl_compare/",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _eval_on_env(model, env, n_episodes: int = 10) -> Dict[str, float]:
    """Run n_episodes and return portfolio metrics (mirrors PortfolioEvalCallback._run_eval)."""
    try:
        ep_len = env.episode_length
    except AttributeError:
        try:
            ep_len = env.unwrapped.episode_length
        except AttributeError:
            ep_len = 63

    episode_returns, portfolio_values, turnovers = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_turnover = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            ep_turnover += info.get("turnover", 0.0)
        episode_returns.append(ep_return)
        portfolio_values.append(info.get("portfolio_value", 1.0))
        turnovers.append(ep_turnover)

    annual_returns = [(pv - 1.0) * (252 / ep_len) for pv in portfolio_values]
    return {
        "eval/episode_return": float(np.mean(episode_returns)),
        "eval/portfolio_value": float(np.mean(portfolio_values)),
        "eval/annual_return": float(np.mean(annual_returns)),
        "eval/avg_turnover": float(np.mean(turnovers)),
        "eval/std_episode_return": float(np.std(episode_returns)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if not FINRL_AVAILABLE:
        raise ImportError("FinRL is not installed. Run: uv sync")

    DRLAgent = _import_drl_agent()

    if args.use_macro:
        print("WARNING: --use_macro is not supported for FinRL-native training. "
              "FinRL uses its own feature pipeline (MACD, RSI, CCI, turbulence). "
              "Use run_sb3.py or run.py with --use_macro for FRED macro features.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Algorithms to train: {args.algos}")

    # ── Download FinRL data (single download, shared across all algos) ───────
    tickers = args.tickers
    if tickers is None:
        tickers = MUTUAL_FUND_TICKERS if args.use_mutual_funds else DEFAULT_TICKERS
    print(f"Tickers: {tickers}")

    # Single download across full date range
    df_all = download_finrl_data(
        tickers=tickers,
        start=args.start_date,
        end=args.end_date,
    )

    # ── Chronological splits ─────────────────────────────────────────────────
    dates = sorted(df_all["date"].unique())
    train_dates = [d for d in dates if args.start_date <= d <= args.train_end]
    val_dates   = [d for d in dates if args.val_start  <= d <= args.val_end]
    test_dates  = [d for d in dates if args.test_start <= d <= args.end_date]

    if not train_dates or not val_dates or not test_dates:
        raise ValueError(
            f"One or more splits are empty. Check date range and ticker data coverage.\n"
            f"  Train: {len(train_dates)} dates, Val: {len(val_dates)}, Test: {len(test_dates)}"
        )

    df_train = df_all[df_all["date"].isin(train_dates)].reset_index(drop=True)
    df_val   = df_all[df_all["date"].isin(val_dates)].reset_index(drop=True)
    df_test  = df_all[df_all["date"].isin(test_dates)].reset_index(drop=True)

    print(f"Train: {train_dates[0]} → {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Val:   {val_dates[0]} → {val_dates[-1]} ({len(val_dates)} days)")
    print(f"Test:  {test_dates[0]} → {test_dates[-1]} ({len(test_dates)} days)")

    # ── Build FinRL PortfolioOptimizationEnv via our wrapper ─────────────────
    from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
        PortfolioOptimizationEnv,
    )
    from src.envs.finrl_wrapper import FinRLPortfolioWrapper
    from src.envs.action_bounded_wrapper import ActionBoundedWrapper

    # Determine feature columns (what actually exists after download)
    base_features = ["close", "high", "low", "macd", "rsi_30", "cci_30", "dx_30",
                     "close_30_sma", "close_60_sma", "turbulence"]
    features = [f for f in base_features if f in df_all.columns]
    print(f"FinRL features: {features}")

    def _make_finrl_env(df_split: pd.DataFrame, bounded: bool = False) -> FinRLPortfolioWrapper:
        raw = PortfolioOptimizationEnv(
            df=df_split,
            initial_amount=100_000,
            comission_fee_pct=args.transaction_cost,
            features=features,
            time_window=args.time_window,
            new_gym_api=True,
            normalize_df="by_previous_time",
        )
        wrapped = FinRLPortfolioWrapper(raw, accept_portfolio_weights=False)
        wrapped.episode_length = len(df_split["date"].unique()) - args.time_window
        if bounded:
            wrapped = ActionBoundedWrapper(wrapped)
        return wrapped

    # ── WandB init (one run per seed covering all algos) ────────────────────
    tickers_actual = sorted(df_all["tic"].unique().tolist())
    run_name = f"finrl_compare_seed{args.seed}"
    wandb.init(
        project="cs285-portfolio-rl",
        group=args.run_group,
        name=run_name,
        config={
            **vars(args),
            "tickers": tickers_actual,
            "n_assets": len(tickers_actual),
            "features": features,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
        },
        sync_tensorboard=True,
    )

    # ── Per-algorithm results accumulator ────────────────────────────────────
    summary_rows = []

    # ── Training loop — one algo at a time ───────────────────────────────────
    for algo in args.algos:
        print(f"\n{'='*60}")
        print(f"  Training: {algo.upper()}  (seed={args.seed})")
        print(f"{'='*60}")

        train_env = _make_finrl_env(df_train, bounded=True)
        val_env   = _make_finrl_env(df_val,   bounded=True)
        test_env  = _make_finrl_env(df_test,  bounded=True)

        obs_dim    = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        print(f"  obs_dim={obs_dim}, action_dim={action_dim}")

        # Build model using DRLAgent (FinRL-native interface)
        drl_agent = DRLAgent(env=train_env)
        model_kwargs = dict(ALGO_MODEL_KWARGS[algo])  # copy so we don't mutate

        # Policy kwargs: shared hidden dim for all algos
        hidden = [256, 256]
        if algo in ("ppo", "a2c"):
            policy_kwargs = dict(net_arch=dict(pi=hidden, vf=hidden))
        else:
            policy_kwargs = dict(net_arch=dict(pi=hidden, qf=hidden))

        model = drl_agent.get_model(
            algo,
            policy="MlpPolicy",
            policy_kwargs=policy_kwargs,
            model_kwargs=model_kwargs,
            verbose=0,
        )

        # Use SB3's learn() directly (DRLAgent.train_model doesn't accept callbacks)
        # so we get WandB + portfolio eval while still using FinRL's get_model API.
        wandb_cb = WandbCallback(gradient_save_freq=0, verbose=0)
        eval_cb  = PortfolioEvalCallback(
            eval_env=val_env,           # evaluate on val split during training
            eval_interval=args.eval_interval,
            n_eval_episodes=args.n_eval_episodes,
            verbose=1,
        )

        # Prefix all WandB keys with algo name so they don't collide in the same run
        _orig_on_step = eval_cb._on_step
        def _patched_on_step(self=eval_cb, algo=algo):
            if self.num_timesteps - self._last_eval_step >= self.eval_interval:
                self._last_eval_step = self.num_timesteps
                metrics = self._run_eval()
                wandb.log(
                    {f"{algo}/{k}": v for k, v in metrics.items()},
                    step=self.num_timesteps,
                )
                if self.verbose:
                    ar = metrics.get("eval/annual_return", 0.0)
                    print(f"  [{algo} val @ {self.num_timesteps}] annual_return={ar:.2%}")
            return True
        eval_cb._on_step = _patched_on_step

        print(f"  Training for {args.total_timesteps:,} steps...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[wandb_cb, eval_cb],
            progress_bar=True,
        )

        # ── Evaluate on val and test splits ──────────────────────────────────
        print(f"  Evaluating on val split...")
        val_metrics  = _eval_on_env(model, val_env,  n_episodes=args.n_eval_episodes)
        print(f"  Evaluating on test split (2022+ OOD regime)...")
        test_metrics = _eval_on_env(model, test_env, n_episodes=20)

        wandb.log(
            {f"{algo}/val/{k.replace('eval/', '')}": v for k, v in val_metrics.items()}
            | {f"{algo}/test/{k.replace('eval/', '')}": v for k, v in test_metrics.items()}
        )

        print(f"  [{algo}] val  annual_return={val_metrics['eval/annual_return']:.2%}  "
              f"portfolio_value={val_metrics['eval/portfolio_value']:.4f}")
        print(f"  [{algo}] test annual_return={test_metrics['eval/annual_return']:.2%}  "
              f"portfolio_value={test_metrics['eval/portfolio_value']:.4f}")

        summary_rows.append({
            "algo": algo,
            "seed": args.seed,
            "val_annual_return":     val_metrics["eval/annual_return"],
            "val_portfolio_value":   val_metrics["eval/portfolio_value"],
            "test_annual_return":    test_metrics["eval/annual_return"],
            "test_portfolio_value":  test_metrics["eval/portfolio_value"],
            "test_avg_turnover":     test_metrics["eval/avg_turnover"],
        })

        # ── Save model checkpoint ─────────────────────────────────────────────
        if args.save_models:
            save_dir = f"results/finrl_compare/{run_name}"
            os.makedirs(save_dir, exist_ok=True)
            model.save(f"{save_dir}/{algo}_model")
            print(f"  Saved: results/finrl_compare/{run_name}/{algo}_model.zip")

    # ── Summary table ─────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    print(f"\n{'='*60}")
    print("SUMMARY (test split 2022–2026, OOD regime)")
    print(df_summary.to_string(index=False))

    wandb.log({"summary": wandb.Table(dataframe=df_summary)})

    if args.save_results:
        save_dir = f"results/finrl_compare/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        df_summary.to_csv(f"{save_dir}/summary.csv", index=False)
        print(f"Saved: {save_dir}/summary.csv")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
