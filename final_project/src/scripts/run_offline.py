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


TRADING_DAYS_PER_YEAR = 252


def compute_sharpe_ratio(simple_returns: np.ndarray, rf_daily: float = 0.0) -> float:
    """
    Annualized Sharpe ratio from per-step *simple* returns.

    Uses unbiased std (ddof=1) and subtracts the daily risk-free rate. Callers
    should pass a single concatenated return stream, not per-episode Sharpes,
    to avoid the small-sample-per-episode bias that inflates the score.
    """
    arr = np.asarray(simple_returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    excess = arr - rf_daily
    mu = excess.mean()
    sig = excess.std(ddof=1)
    if sig < 1e-12:
        return 0.0
    return float(mu / sig * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_max_drawdown(portfolio_values: list) -> float:
    """Compute maximum drawdown from a sequence of portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0
    arr = np.array(portfolio_values, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (running_max - arr) / np.maximum(running_max, 1e-8)
    return float(drawdowns.max())


@torch.no_grad()
def evaluate_agent(agent, env, n_episodes, device, rf_daily: float = 0.0):
    """
    Unified evaluation for all offline agents.

    Returns per-split metrics under the ``eval/`` namespace. The caller is
    responsible for renaming the namespace (e.g. to ``final_train/``) when
    logging multiple splits side by side.

    Notes on the math:
      * ``episode_return``  — mean cumulative log-return across episodes.
      * ``portfolio_value`` — mean final PV across episodes.
      * ``annual_return``   — geometric annualization of per-episode PV.
      * ``sharpe_ratio``    — computed once on the concatenated per-step
        simple-return stream across all eval episodes. Averaging per-episode
        Sharpes over 63-step windows is biased high and is not a meaningful
        financial quantity.
      * ``avg_turnover``    — MEAN per-step L1 turnover (previously a per-episode
        sum, which hid whether the agent was actually rebalancing).
    """
    episode_log_returns = []
    portfolio_values = []
    episode_total_turnover = []
    all_step_simple_returns = []
    all_max_drawdowns = []
    n_steps_total = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_log_return = 0.0
        ep_turnover = 0.0
        pv_trajectory = [1.0]
        info = {}

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.get_action(obs_t)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Prefer the env-reported log/simple returns if present; fall back
            # to the reward scalar (which equals the log return for
            # reward_type='log_return' in PortfolioEnv after the return fix).
            step_log = float(info.get('portfolio_log_return', reward))
            step_simple = float(info.get('portfolio_return', np.expm1(reward)))
            ep_log_return += step_log
            ep_turnover += float(info.get('turnover', 0.0))
            all_step_simple_returns.append(step_simple)
            pv_trajectory.append(float(info.get('portfolio_value', pv_trajectory[-1])))
            n_steps_total += 1

        episode_log_returns.append(ep_log_return)
        portfolio_values.append(float(info.get('portfolio_value', 1.0)))
        episode_total_turnover.append(ep_turnover)
        all_max_drawdowns.append(compute_max_drawdown(pv_trajectory))

    # Geometric annualization — no more linear (pv-1)*252/L shortcut.
    annual_returns = [
        pv ** (TRADING_DAYS_PER_YEAR / env.episode_length) - 1.0
        for pv in portfolio_values
    ]
    # Per-step mean turnover across ALL steps in the eval run.
    avg_step_turnover = (
        float(np.sum(episode_total_turnover) / max(n_steps_total, 1))
        if n_steps_total > 0 else 0.0
    )
    # Single Sharpe over the concatenated simple-return stream.
    sharpe = compute_sharpe_ratio(np.asarray(all_step_simple_returns), rf_daily=rf_daily)

    return {
        'eval/episode_return': float(np.mean(episode_log_returns)),
        'eval/portfolio_value': float(np.mean(portfolio_values)),
        'eval/annual_return': float(np.mean(annual_returns)),
        'eval/avg_turnover': avg_step_turnover,
        'eval/std_episode_return': float(np.std(episode_log_returns, ddof=1))
            if len(episode_log_returns) > 1 else 0.0,
        'eval/sharpe_ratio': sharpe,
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
        # Every offline agent in this repo outputs weights on the simplex
        # (Dirichlet mean / softmax head), so the env must NOT apply another
        # softmax to them — that bug silently collapses the policy to uniform.
        accept_portfolio_weights=True,
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

    # Final evaluation on all three splits so the comparison is apples-to-apples.
    n_final = args.n_eval_episodes * 2
    final_train = evaluate_agent(agent, train_env, n_final, device)
    final_val   = evaluate_agent(agent, val_env,   n_final, device)
    final_test  = evaluate_agent(agent, test_env,  n_final, device)

    def _rename(metrics: dict, prefix: str) -> dict:
        return {f"{prefix}/{k.split('/')[-1]}": v for k, v in metrics.items()}

    wandb.log({
        **_rename(final_train, "final_train"),
        **_rename(final_val,   "final_val"),
        **_rename(final_test,  "final_test"),
        # Keep the legacy 'test/...' keys so old dashboards still work.
        **_rename(final_test,  "test"),
    })

    def _fmt(m: dict) -> str:
        return (
            f"ret={m['eval/episode_return']:+.4f}  "
            f"pv={m['eval/portfolio_value']:.4f}  "
            f"annret={m['eval/annual_return']:+.4f}  "
            f"sharpe={m['eval/sharpe_ratio']:+.3f}  "
            f"mdd={m['eval/max_drawdown']:.4f}  "
            f"turnover={m['eval/avg_turnover']:.4f}"
        )

    print("\n=== Final evaluation ===")
    print(f"Train : {_fmt(final_train)}")
    print(f"Val   : {_fmt(final_val)}")
    print(f"Test  : {_fmt(final_test)}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
