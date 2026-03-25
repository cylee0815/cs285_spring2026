"""
Training entry point for the O2O pipeline.

Chronological split (aligned to proposal's distribution-shift experiment):
  Train  (offline):  2008-01-01 → 2020-12-31  (GFC 2008 + COVID 2020)
  Val    (HP tuning): 2021-01-01 → 2021-12-31
  Test   (O2O online): 2022-01-01 → 2026-03-31  (stock/bond correlation break)

Usage:
    # Standard ETF universe (SPY, EEM, TLT, HYG, DBC, GLD, UUP, SHY — data from 2008)
    uv run src/scripts/run_o2o.py --run_group=debug --phase=offline --seed=0
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=o2o --seed=0

    # Mutual fund proxies — extends history to 1990s (Dot-Com bubble coverage)
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=o2o --use_mutual_funds --start_date=1995-01-01 --seed=0

    # FinRL env for online phase (richer features: MACD, RSI, CCI, turbulence)
    # Note: offline phase always uses custom env (FinRL env resets from start of data,
    #       not suitable for random-start offline trajectory generation)
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=sac --use_finrl --seed=0
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=o2o --use_finrl_online --seed=0
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
    make_train_test_envs, make_train_val_test_envs, make_train_test_envs_finrl,
    DEFAULT_TICKERS, MUTUAL_FUND_TICKERS,
)
from src.agents.cql_geodesic import GeodesicCQL
from src.agents.sac_dirichlet import SACDirichlet
from src.agents.o2o_agent import O2OAgent
from src.agents.replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", type=str, default="debug")
    parser.add_argument("--phase", type=str, default="o2o",
                        choices=["offline", "online", "o2o", "sac"],
                        help="offline=CQL only, online=SAC only, o2o=full pipeline, sac=online SAC baseline")
    parser.add_argument("--seed", type=int, default=0)
    # Environment — ticker universe
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list (overrides --use_mutual_funds).")
    parser.add_argument(
        "--use_mutual_funds",
        action="store_true",
        help="Use mutual fund proxies (VFINX, VUSTX, etc.) instead of standard ETFs. "
             "Extends history back to the 1990s for Dot-Com bubble coverage.",
    )
    # Chronological split dates
    parser.add_argument("--start_date", type=str, default="2008-01-01",
                        help="Start of training data. Use 1995-01-01 with --use_mutual_funds.")
    parser.add_argument("--train_end",  type=str, default="2020-12-31")
    parser.add_argument("--val_start",  type=str, default="2021-01-01")
    parser.add_argument("--val_end",    type=str, default="2021-12-31")
    parser.add_argument("--test_start", type=str, default="2022-01-01")
    parser.add_argument("--end_date",   type=str, default="2026-03-31",
                        help="End of test data.")
    parser.add_argument("--episode_length", type=int, default=63)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--reward_type", type=str, default="log_return")
    # FinRL options
    parser.add_argument(
        "--use_finrl_online",
        action="store_true",
        help="Use FinRL env for the online fine-tuning phase (SAC/O2O). "
             "Offline phase always uses the custom env regardless of this flag.",
    )
    parser.add_argument(
        "--use_finrl",
        action="store_true",
        help="Shorthand: use FinRL env for online phase (same as --use_finrl_online).",
    )
    parser.add_argument("--finrl_time_window", type=int, default=20)
    # Multimodal feature flags (hypothesis 6: multimodal information advantage)
    parser.add_argument(
        "--use_macro", action="store_true",
        help="Append 8 FRED macroeconomic features (rates, CPI, unemployment, GDP, sentiment). "
             "Requires FRED_API_KEY env var (free key at fred.stlouisfed.org). Falls back to zeros.",
    )
    parser.add_argument(
        "--use_sentiment", action="store_true",
        help="Append SF Fed Daily News Sentiment Index (auto-downloaded on first run).",
    )
    parser.add_argument(
        "--use_alpaca_embeddings", action="store_true",
        help="Append Alpaca News sentence embeddings (384-d). Requires precomputed cache; "
             "see src/envs/sentiment_features.AlpacaNewsEmbeddings.",
    )
    # Training scale
    parser.add_argument("--n_offline_updates", type=int, default=None)
    parser.add_argument("--n_online_steps", type=int, default=None)
    parser.add_argument("--offline_data_steps", type=int, default=50_000)
    parser.add_argument("--eval_interval", type=int, default=5_000)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_finrl_online = args.use_finrl_online or args.use_finrl

    # ── 3-way chronological split (offline=train, val=HP tuning, online=test) ──
    # Offline phase always uses the custom env (random-start episode sampling required).
    tickers = args.tickers  # None → make_train_val_test_envs selects based on use_mutual_funds
    ticker_label = (
        "mutual_fund_proxies" if (args.use_mutual_funds and tickers is None)
        else str(tickers or DEFAULT_TICKERS)
    )
    print(f"Downloading market data: {ticker_label}")

    custom_train_env, custom_val_env, custom_test_env, metadata = make_train_val_test_envs(
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

    # Online phase: optionally use FinRL env for richer observations
    if use_finrl_online and args.phase in ("sac", "o2o", "online"):
        print("  [Online] Using FinRL environment (test split)")
        # DirichletActor outputs portfolio weights → accept_portfolio_weights=True
        online_train_env, online_test_env, finrl_meta = make_train_test_envs_finrl(
            tickers=metadata["tickers"],
            start=args.test_start,
            end=args.end_date,
            time_window=args.finrl_time_window,
            transaction_cost=args.transaction_cost,
            accept_portfolio_weights=True,  # DirichletActor: weights → log(w) → softmax → w
        )
        metadata.update({k: v for k, v in finrl_meta.items() if k not in metadata})
        metadata["env_backend"] = "finrl"
        print(f"  [FinRL] obs_dim={finrl_meta['obs_dim']}, action_dim={finrl_meta['action_dim']}")
    else:
        online_train_env, online_test_env = custom_test_env, custom_test_env
        metadata["env_backend"] = "custom"

    train_env = custom_train_env   # offline pre-training (2005–2020)
    val_env   = custom_val_env     # hyperparameter evaluation (2021)
    test_env  = online_test_env    # O2O fine-tuning & final eval (2022+)

    print(f"Train (offline): {metadata['train_start']} → {metadata['train_end']}  ({metadata['T_train']} days)")
    print(f"Val   (HP eval): {metadata['val_start']} → {metadata['val_end']}  ({metadata['T_val']} days)")
    print(f"Test  (online):  {metadata['test_start']} → {metadata['test_end']}  ({metadata['T_test']} days)")

    # Load config
    if args.phase in ["offline", "o2o"]:
        from src.configs.o2o_config import get_config
    else:
        from src.configs.sac_dirichlet_config import get_config
    config = get_config()

    # Apply CLI overrides
    if args.n_offline_updates:
        config.n_offline_updates = args.n_offline_updates
    if args.n_online_steps:
        config.n_online_steps = args.n_online_steps

    # Init WandB
    run_name = f"{args.phase}_seed{args.seed}"
    wandb.init(
        project="cs285-portfolio-rl",
        group=args.run_group,
        name=run_name,
        config={**dict(config), **vars(args), **metadata},
    )

    # --- Full O2O pipeline ---
    if args.phase == "o2o":
        # Offline phase uses train env (2005–2020); online phase uses test env (2022+)
        agent = O2OAgent(custom_train_env, test_env, config, device)
        # Swap the SAC agent's env to the online env (FinRL or custom test split)
        agent.sac_agent.env = online_train_env

        # Load offline data
        agent.load_offline_data(n_steps=args.offline_data_steps)

        # Phase 1: offline pre-training — evaluate on val split (2021) to avoid look-ahead bias
        print("\n=== Phase 1: Offline Geodesic-CQL Pre-training ===")
        n_offline = config.n_offline_updates
        for step in trange(n_offline, desc="Offline"):
            metrics = agent.cql_agent.update()
            if step % args.eval_interval == 0:
                eval_metrics = agent._evaluate_cql(n_episodes=5)
                wandb.log({**{f"offline/{k}": v for k, v in metrics.items()},
                           **eval_metrics, "step": step})

        # Phase 2: online fine-tuning on test split (2022+)
        print("\n=== Phase 2: Online SAC-Dirichlet Fine-tuning ===")
        agent.transfer_to_online()
        n_online = config.n_online_steps
        for step in trange(n_online, desc="Online"):
            agent.sac_agent.collect_step()
            if len(agent.sac_agent.buffer) >= config.batch_size:
                online_batch = agent.sac_agent.buffer.sample_with_context(config.batch_size // 2)
                offline_batch = agent.offline_buffer.sample_with_context(config.batch_size // 2)
                mixed = {k: torch.cat([online_batch[k], offline_batch[k]], dim=0)
                         for k in online_batch if k in offline_batch}
                m = {}
                m.update(agent.sac_agent.update_critic(mixed))
                m.update(agent.sac_agent.update_actor(mixed))
                m.update(agent.sac_agent.update_temperature(mixed))
                agent.sac_agent.update_target_critic()
                if step % args.eval_interval == 0:
                    eval_metrics = agent.sac_agent.evaluate(test_env, n_episodes=5)
                    wandb.log({**{f"online/{k}": v for k, v in m.items()},
                               **eval_metrics, "step": n_offline + step})

    # --- Online SAC-Dirichlet only (baseline) ---
    elif args.phase == "sac":
        agent = SACDirichlet(online_train_env, config, device)
        n_steps = config.get("n_online_steps", 200_000)
        for step in trange(n_steps, desc="SAC-Dirichlet"):
            metrics = agent.update()
            if step % args.eval_interval == 0 and metrics:
                eval_metrics = agent.evaluate(test_env, n_episodes=5)
                wandb.log({**{f"train/{k}": v for k, v in metrics.items()},
                           **eval_metrics, "step": step})

    # --- Offline only ---
    elif args.phase == "offline":
        obs_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        offline_buffer = ReplayBuffer(config.offline_buffer_size, obs_dim, action_dim, device,
                                      seq_len=config.regime_window)
        offline_buffer.load_from_env(train_env, n_steps=args.offline_data_steps)
        offline_buffer.freeze()
        agent = GeodesicCQL(obs_dim, action_dim, config, device, offline_buffer=offline_buffer)
        n_steps = config.n_offline_updates
        for step in trange(n_steps, desc="Geodesic-CQL"):
            metrics = agent.update()
            if step % args.eval_interval == 0:
                # Evaluate on val split (2021) during offline training
                from src.agents.o2o_agent import O2OAgent
                tmp = O2OAgent.__new__(O2OAgent)
                tmp.eval_env = val_env
                tmp.cql_agent = agent
                tmp.device = device
                eval_metrics = tmp._evaluate_cql(n_episodes=5)
                wandb.log({**{f"offline/{k}": v for k, v in metrics.items()},
                           **eval_metrics, "step": step})

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
