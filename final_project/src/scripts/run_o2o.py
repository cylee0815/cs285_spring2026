"""
Training entry point for the O2O pipeline.

Usage:
    uv run src/scripts/run_o2o.py --run_group=debug --phase=offline --seed=0
    uv run src/scripts/run_o2o.py --run_group=debug --phase=online --seed=0
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=o2o --seed=0  # full pipeline
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=sac --seed=0  # online-only baseline
"""
import argparse
import importlib
import random
import numpy as np
import torch
import wandb
from tqdm import trange

from src.envs.data_utils import make_train_test_envs, DEFAULT_TICKERS
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
    # Environment
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start_date", type=str, default="2010-01-01")
    parser.add_argument("--end_date", type=str, default="2024-01-01")
    parser.add_argument("--episode_length", type=int, default=63)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--reward_type", type=str, default="log_return")
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

    # Build environments
    tickers = args.tickers or DEFAULT_TICKERS
    print(f"Downloading data for: {tickers}")
    train_env, test_env, metadata = make_train_test_envs(
        tickers=tickers,
        start=args.start_date,
        end=args.end_date,
        episode_length=args.episode_length,
        transaction_cost=args.transaction_cost,
        reward_type=args.reward_type,
    )
    print(f"Train: {metadata['train_start']} → {metadata['train_end']}")
    print(f"Test:  {metadata['test_start']} → {metadata['test_end']}")

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
        agent = O2OAgent(train_env, test_env, config, device)

        # Load offline data
        agent.load_offline_data(n_steps=args.offline_data_steps)

        # Phase 1: offline pre-training
        print("\n=== Phase 1: Offline Geodesic-CQL Pre-training ===")
        n_offline = config.n_offline_updates
        for step in trange(n_offline, desc="Offline"):
            metrics = agent.cql_agent.update()
            if step % args.eval_interval == 0:
                eval_metrics = agent._evaluate_cql(n_episodes=5)
                wandb.log({**{f"offline/{k}": v for k, v in metrics.items()},
                           **eval_metrics, "step": step})

        # Phase 2: online fine-tuning
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
        agent = SACDirichlet(train_env, config, device)
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
                # Inline eval
                from src.agents.o2o_agent import O2OAgent
                tmp = O2OAgent.__new__(O2OAgent)
                tmp.eval_env = test_env
                tmp.cql_agent = agent
                tmp.device = device
                eval_metrics = tmp._evaluate_cql(n_episodes=5)
                wandb.log({**{f"offline/{k}": v for k, v in metrics.items()},
                           **eval_metrics, "step": step})

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
