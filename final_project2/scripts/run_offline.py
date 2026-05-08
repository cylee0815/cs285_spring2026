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
import copy
import importlib
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import trange

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.envs.data_utils import (
    make_train_val_test_envs, DEFAULT_TICKERS, MUTUAL_FUND_TICKERS,
)
from core.envs.portfolio_obs_wrapper import PortfolioObsWrapper
from core.buffers.replay_buffer import ReplayBuffer, NStepReplayBuffer
from offline_rl.configs import CONFIG_MAP


AGENT_MAP = {
    "bc": ("offline_rl.agents.bc", "BCAgent"),
    "fisher_bc": ("offline_rl.agents.fisher_bc", "FisherBCAgent"),
    "td3_bc": ("offline_rl.agents.td3_bc", "TD3BCAgent"),
    "awac": ("offline_rl.agents.awac", "AWACAgent"),
    "cql_vanilla": ("offline_rl.agents.cql_vanilla", "VanillaCQLAgent"),
    "iql": ("offline_rl.agents.iql", "IQLAgent"),
    "edac": ("offline_rl.agents.edac", "EDACAgent"),
    "bcq": ("offline_rl.agents.bcq", "BCQAgent"),
    "mbpo": ("offline_rl.agents.mbpo", "MBPOAgent"),
    "mopo": ("offline_rl.agents.mopo", "MOPOAgent"),
    "decision_transformer": ("offline_rl.agents.decision_transformer", "DecisionTransformerAgent"),
    "trajectory_transformer": ("offline_rl.agents.trajectory_transformer", "TrajectoryTransformerAgent"),
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
    # Smoke-test / safety flags. ``--smoke_test`` forces tiny updates + tiny
    # offline buffer + no WandB, so CI / local dev can exercise the full
    # training loop in a few seconds without ever kicking off a long run.
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a tiny end-to-end sanity check (20 updates, "
                             "256-step buffer, no WandB). Overrides --n_offline_updates "
                             "and --offline_data_steps when set.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging (also implied by --smoke_test).")
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
def evaluate_agent(
    agent,
    env,
    n_episodes,
    device,
    rf_daily: float = 0.0,
    *,
    ticker_names=None,
    log_step=None,
    split_name: str = "val",
    save_csv: bool = False,
    csv_path=None,
):
    """
    Chronological-backtest evaluation for offline agents.

    Evaluation is a SINGLE trajectory that sweeps the env's dataset from
    day 0 to the end of the split. The ``n_episodes`` argument is ignored
    and forced to 1 — randomized multi-episode evaluation would re-shuffle
    the backtest start and make the Sharpe / drawdown metrics incomparable
    across splits.

    Observability hooks:
      * ``log_step`` (int | None): if provided AND a wandb run is active,
        logs the PV trajectory as a ``wandb.plot.line`` chart under the
        unique key ``"{split_name}_trajectory_step_{log_step}"``.
      * ``save_csv`` + ``csv_path``: if True, writes a per-step trade log
        (Step, Portfolio_Value, Step_Reward, w_<ticker>…) to ``csv_path``.
      * ``ticker_names`` (list[str] | None): column labels for the weight
        columns in the CSV; falls back to ``w_0, w_1, …``.

    Metrics (under the ``eval/`` namespace):
      * ``episode_return``      — cumulative log-return over the full sweep.
      * ``portfolio_value``     — final PV at end of sweep.
      * ``annual_return``       — geometric annualization of the final PV.
      * ``annual_volatility``   — std(simple_returns) * sqrt(252).
      * ``sharpe_ratio``        — annualized, on simple-return stream.
      * ``sortino_ratio``       — annualized, downside-deviation denominator.
      * ``calmar_ratio``        — annual_return / |max_drawdown|.
      * ``max_drawdown``        — worst peak-to-trough drawdown over sweep.
      * ``win_rate``            — fraction of days with simple return > 0.
      * ``avg_turnover``        — mean per-step L1 turnover.
    """
    # Force chronological single-sweep backtest. Any caller-supplied
    # ``n_episodes`` is intentionally overridden.
    n_episodes = 1

    episode_log_returns = []
    portfolio_values = []
    episode_total_turnover = []
    all_step_simple_returns = []
    all_max_drawdowns = []
    # Per-step trade log for CSV export (collected across the single sweep).
    trade_rows = []
    n_steps_total = 0

    for _ in range(n_episodes):
        # randomize=False ensures the episode starts at day 0 of the split
        # and runs to env.episode_length — giving a single deterministic
        # chronological backtest.
        obs, _ = env.reset(options={"randomize": False})
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
            step_pv = float(info.get('portfolio_value', pv_trajectory[-1]))
            pv_trajectory.append(step_pv)

            # Record per-step trade row — raw action output by the agent is
            # what the user asked to verify; with accept_portfolio_weights=True
            # these are already normalized simplex weights.
            action_np = np.asarray(action, dtype=np.float64).flatten()
            trade_rows.append({
                'Step': n_steps_total,
                'Date': info.get('date', ''),
                'Portfolio_Value': step_pv,
                'Step_Reward': float(reward),
                '_action': action_np,
            })
            n_steps_total += 1

        episode_log_returns.append(ep_log_return)
        portfolio_values.append(float(info.get('portfolio_value', 1.0)))
        episode_total_turnover.append(ep_turnover)
        all_max_drawdowns.append(compute_max_drawdown(pv_trajectory))

    # Geometric annualization of the final PV.
    annual_returns = [
        pv ** (TRADING_DAYS_PER_YEAR / max(env.episode_length, 1)) - 1.0
        for pv in portfolio_values
    ]
    # Per-step mean turnover across ALL steps in the eval run.
    avg_step_turnover = (
        float(np.sum(episode_total_turnover) / max(n_steps_total, 1))
        if n_steps_total > 0 else 0.0
    )

    simple_arr = np.asarray(all_step_simple_returns, dtype=np.float64)
    # Single Sharpe over the concatenated simple-return stream.
    sharpe = compute_sharpe_ratio(simple_arr, rf_daily=rf_daily)

    # Annualized volatility: std(daily simple returns) * sqrt(252).
    if simple_arr.size > 1:
        annual_vol = float(simple_arr.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        annual_vol = 0.0

    # Sortino: mean excess return / downside deviation, annualized.
    # Downside deviation = std of strictly-negative excess returns.
    excess = simple_arr - rf_daily
    negative = excess[excess < 0.0]
    if negative.size > 1:
        downside_dev = float(negative.std(ddof=1))
        if downside_dev > 1e-12:
            sortino = float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR))
        else:
            sortino = 0.0
    else:
        sortino = 0.0

    # Calmar: annualized return / |max drawdown|. Guard against div-by-zero.
    mean_annual_return = float(np.mean(annual_returns)) if annual_returns else 0.0
    mean_max_dd = float(np.mean(all_max_drawdowns)) if all_max_drawdowns else 0.0
    calmar = mean_annual_return / abs(mean_max_dd) if abs(mean_max_dd) > 1e-12 else 0.0

    # Win rate: fraction of days with strictly positive simple return.
    win_rate = float(np.mean(simple_arr > 0.0)) if simple_arr.size > 0 else 0.0

    # ── WandB: log full PV trajectory as a Matplotlib image ─────────────
    # Matplotlib gives us a dynamically-scaled y-axis (so drawdowns and
    # run-ups are visible) and a thicker, high-contrast line, unlike the
    # default ``wandb.plot.line`` chart which is faint and pinned to y=0.
    if log_step is not None and wandb.run is not None and len(pv_trajectory) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pv_trajectory, color="blue", linewidth=2)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value")
        ax.set_title(f"{split_name} trajectory @ train step {log_step}")
        ax.grid(True, alpha=0.3)
        plot_key = f"{split_name}_trajectory_step_{log_step}"
        wandb.log({plot_key: wandb.Image(fig)})
        plt.close(fig)

    # ── Trade history CSV export ────────────────────────────────────────
    if save_csv and csv_path and trade_rows:
        n_weights = int(trade_rows[0]['_action'].size)
        if ticker_names is not None and len(ticker_names) >= n_weights:
            weight_cols = [f"w_{t}" for t in ticker_names[:n_weights]]
        else:
            weight_cols = [f"w_{i}" for i in range(n_weights)]

        df_rows = []
        for r in trade_rows:
            row = {
                'Step': r['Step'],
                'Date': r['Date'],
                'Portfolio_Value': r['Portfolio_Value'],
                'Step_Reward': r['Step_Reward'],
            }
            for col, w in zip(weight_cols, r['_action']):
                row[col] = float(w)
            df_rows.append(row)
        df = pd.DataFrame(df_rows)

        out_dir = os.path.dirname(csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)

    return {
        'eval/episode_return': float(np.mean(episode_log_returns)),
        'eval/portfolio_value': float(np.mean(portfolio_values)),
        'eval/annual_return': mean_annual_return,
        'eval/annual_volatility': annual_vol,
        'eval/avg_turnover': avg_step_turnover,
        'eval/sharpe_ratio': sharpe,
        'eval/sortino_ratio': sortino,
        'eval/calmar_ratio': calmar,
        'eval/max_drawdown': mean_max_dd,
        'eval/win_rate': win_rate,
    }


def main():
    args = parse_args()

    # Task 3: MBPO/MOPO require weight-aware dynamics models that are
    # incompatible with the current observation wrapper.  Disable them until
    # the dynamics model is updated to handle the augmented obs space.
    if args.base_config in ("mbpo", "mopo"):
        raise NotImplementedError(
            "MBPO/MOPO require weight-aware observations; currently disabled."
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Smoke-test: date-range + episode-length overrides ───────────────
    # These have to run BEFORE env construction since they feed into
    # ``make_train_val_test_envs``. The config-dependent overrides
    # (updates, batch_size, buffer size) run after the config is loaded.
    if args.smoke_test:
        args.start_date = "2022-01-01"
        args.train_end = "2022-12-31"
        args.val_start = "2023-01-01"
        args.val_end = "2023-06-30"
        args.test_start = "2023-07-01"
        args.end_date = "2023-12-31"
        args.episode_length = 20
        args.offline_data_steps = min(args.offline_data_steps, 256)
        args.n_eval_episodes = 1
        args.eval_interval = 10

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

    # Wrap ALL envs so observations include previous portfolio weights.
    # This must happen BEFORE episode_length overrides since the wrapper
    # delegates attribute access to the underlying env.
    train_env = PortfolioObsWrapper(train_env)
    val_env   = PortfolioObsWrapper(val_env)
    test_env  = PortfolioObsWrapper(test_env)

    # Full-horizon evaluation envs. ``train_env`` keeps its 63-day window so
    # the offline replay buffer still samples random short episodes; the
    # validation and test envs run their ENTIRE split as one chronological
    # backtest. ``eval_train_env`` is a shallow copy of train_env so we can
    # also run a full-train backtest at final eval time without disturbing
    # the buffer-loading env.
    # Set full-horizon episode lengths on the underlying PortfolioEnv so that
    # reset()/step() see the right limit. The wrapper's __getattr__ will
    # also forward reads of .episode_length to the inner env.
    val_env.env.episode_length = metadata['T_val']
    test_env.env.episode_length = metadata['T_test']
    eval_train_base = copy.copy(train_env.env)
    eval_train_base.episode_length = metadata['T_train']
    eval_train_env = PortfolioObsWrapper(eval_train_base)

    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    print(f"Train (offline): {metadata['train_start']} -> {metadata['train_end']}  ({metadata['T_train']} days)")
    print(f"Val   (HP eval): {metadata['val_start']} -> {metadata['val_end']}  ({metadata['T_val']} days)")
    print(f"Test  (eval):    {metadata['test_start']} -> {metadata['test_end']}  ({metadata['T_test']} days)")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")

    # Load config
    config_module = importlib.import_module(f"offline_rl.configs.{args.base_config}_config")
    config = config_module.get_config()
    if args.n_offline_updates:
        config.n_offline_updates = args.n_offline_updates
    # Always propagate n_step so every agent can compute gamma^n_step.
    config.n_step = args.n_step

    # ── Smoke-test overrides ────────────────────────────────────────────
    # Keep the full pipeline intact but clip it aggressively. Anything that
    # depends on config.* is patched here so agents that read e.g.
    # config.batch_size or config.eval_interval still pick up the small
    # values.
    if args.smoke_test:
        config.n_offline_updates = 20
        if hasattr(config, "batch_size"):
            config.batch_size = min(int(config.batch_size), 32)
        if hasattr(config, "offline_buffer_size"):
            config.offline_buffer_size = 512
        print("[smoke_test] config overrides applied: "
              f"updates={config.n_offline_updates}, "
              f"batch_size={getattr(config, 'batch_size', 'n/a')}, "
              f"buffer_size={getattr(config, 'offline_buffer_size', 'n/a')}")

    # Init WandB (disabled entirely for smoke tests or when --no_wandb is set).
    run_name = f"{args.base_config}_seed{args.seed}"
    use_wandb = not (args.smoke_test or args.no_wandb)
    if use_wandb:
        wandb.init(
            project="cs285-portfolio-rl",
            group=args.run_group,
            name=run_name,
            config={**dict(config), **vars(args), **metadata},
        )
    else:
        # wandb.run is None by default; evaluate_agent already guards on that.
        # We still want wandb.log() to be a no-op so we don't crash paths that
        # log unconditionally.
        wandb.init = lambda *a, **k: None  # type: ignore[assignment]
        wandb.log = lambda *a, **k: None   # type: ignore[assignment]
        wandb.finish = lambda *a, **k: None  # type: ignore[assignment]

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

    # Observability outputs: per-step trade logs are written to a local
    # directory (one CSV per validation phase + one per final split).
    trades_dir = os.path.join("results", "trades", run_name)
    os.makedirs(trades_dir, exist_ok=True)
    eval_tickers = metadata.get('tickers')

    # Training loop — intermediate evaluation is strictly on the VALIDATION
    # split. The test set is held out until the single final evaluation
    # after training completes, to prevent data leakage / test-set HP tuning.
    n_updates = config.n_offline_updates
    for step in trange(n_updates, desc=args.base_config):
        metrics = agent.update()
        if step % args.eval_interval == 0 and metrics:
            val_csv = os.path.join(trades_dir, f"val_trades_step_{step}.csv")
            eval_metrics = evaluate_agent(
                agent, val_env, args.n_eval_episodes, device,
                ticker_names=eval_tickers,
                log_step=step,
                split_name="val",
                save_csv=True,
                csv_path=val_csv,
            )
            wandb.log({
                **{f"train/{k}": v for k, v in metrics.items()},
                **eval_metrics,
                "step": step,
            })

    # Final chronological backtest on all three splits. Each call runs a
    # single full-sweep episode (evaluate_agent forces n_episodes=1 and
    # passes randomize=False). This is the ONE time we touch the test set.
    final_train = evaluate_agent(
        agent, eval_train_env, 1, device,
        ticker_names=eval_tickers,
        log_step=n_updates,
        split_name="final_train",
        save_csv=True,
        csv_path=os.path.join(trades_dir, "final_train_trades.csv"),
    )
    final_val = evaluate_agent(
        agent, val_env, 1, device,
        ticker_names=eval_tickers,
        log_step=n_updates,
        split_name="final_val",
        save_csv=True,
        csv_path=os.path.join(trades_dir, "final_val_trades.csv"),
    )
    final_test = evaluate_agent(
        agent, test_env, 1, device,
        ticker_names=eval_tickers,
        log_step=n_updates,
        split_name="final_test",
        save_csv=True,
        csv_path=os.path.join(trades_dir, "final_test_trades.csv"),
    )

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
            f"annvol={m['eval/annual_volatility']:.4f}  "
            f"sharpe={m['eval/sharpe_ratio']:+.3f}  "
            f"sortino={m['eval/sortino_ratio']:+.3f}  "
            f"calmar={m['eval/calmar_ratio']:+.3f}  "
            f"mdd={m['eval/max_drawdown']:.4f}  "
            f"win={m['eval/win_rate']:.3f}  "
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
